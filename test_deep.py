import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from torch.amp import GradScaler, autocast

# 定义 DeepLabv3+ 的 ASPP 模块
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=64, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.aspp1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.aspp2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[0], dilation=atrous_rates[0], bias=False)
        self.aspp3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[1], dilation=atrous_rates[1], bias=False)
        self.aspp4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rates[2], dilation=atrous_rates[2], bias=False)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.conv1x1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 定义 DeepLabv3+ 解码器
class DeepLabDecoder(nn.Module):
    def __init__(self, low_level_channels, num_classes, aspp_out_channels=64):
        super(DeepLabDecoder, self).__init__()
        self.low_level_conv = nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False)
        self.low_level_bn = nn.BatchNorm2d(48)
        self.low_level_relu = nn.ReLU()
        
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(aspp_out_channels + 48, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x, low_level_feat):
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=False)
        low_level_feat = self.low_level_conv(low_level_feat)
        low_level_feat = self.low_level_bn(low_level_feat)
        low_level_feat = self.low_level_relu(low_level_feat)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.decoder_conv(x)
        return x

# 修改后的 UltraLightSegmentation 模型
class UltraLightSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(UltraLightSegmentation, self).__init__()
        print("Initializing UltraLightSegmentation model with MobileNetV3-Small and DeepLabv3+ Decoder")
        self.backbone = mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.aspp = ASPP(in_channels=576, out_channels=64)  # MobileNetV3-Small 输出通道数为 576，out_channels要和aspp_out_channels=64一致
        self.decoder = DeepLabDecoder(low_level_channels=16, num_classes=num_classes)  # 低层特征来自 conv1 (16 channels)

    def forward(self, x):
        input_size = x.size()[2:]  # 保存输入分辨率 (512, 640)
        low_level_feat = self.backbone.features[0](x)  # 提取低层特征
        x = self.backbone.features(x)  # 提取高层特征
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x

# 自定义数据集类（保持不变）
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        try:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
        except Exception as e:
            print(f"Error loading {img_path} or {mask_path}: {e}")
            raise e
        
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)
        
        mask_np = mask.numpy()
        unique_values = np.unique(mask_np)
        if not np.all(np.isin(unique_values, [0, 1])):
            print(f"Warning: Mask {mask_path} contains invalid values: {unique_values}")
            if mask_np.max() > 1:
                mask_np = (mask_np / 255).astype(np.uint8)
                mask = torch.from_numpy(mask_np)
        
        return image, mask

# 数据预处理（保持不变）
image_transform = transforms.Compose([
    transforms.Resize((512, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 640), interpolation=Image.NEAREST),
    transforms.ToTensor()
])

# 数据集路径（保持不变）
train_image_dir = 'E:/Robotics/Work/cv/dataset/train/images'
train_mask_dir = 'E:/Robotics/Work/cv/dataset/train/masks'
val_image_dir = 'E:/Robotics/Work/cv/dataset/test/images'
val_mask_dir = 'E:/Robotics/Work/cv/dataset/test/masks'

# 加载数据集
train_dataset = SegmentationDataset(train_image_dir, train_mask_dir, transform=image_transform, mask_transform=mask_transform)
val_dataset = SegmentationDataset(val_image_dir, val_mask_dir, transform=image_transform, mask_transform=mask_transform)

# 设置 DataLoader（保持不变）
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

# 主程序
if __name__ == '__main__':
    # 设置多进程启动方式
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # 初始化模型、损失函数和优化器
    model = UltraLightSegmentation(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    scaler = GradScaler()

    # 验证参数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {count_parameters(model)}")

    # 训练循环（保持不变）
    def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
        checkpoint_dir = 'E:/Robotics/Work/cv/codes/deepmodel_smaller'

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, (images, masks) in enumerate(train_loader):
                images, masks = images.to(device), masks.to(device).squeeze(1).long()
                
                optimizer.zero_grad()
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    if epoch == 0 and i == 0:
                        print(f"Epoch 1, Batch 1, Output shape: {outputs.shape}, Mask shape: {masks.shape}")
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item() * images.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
            
            # 验证
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device).squeeze(1).long()
                    with autocast(device_type='cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                    val_loss += loss.item() * images.size(0)
            
            val_loss = val_loss / len(val_loader.dataset)
            print(f'Validation Loss: {val_loss:.4f}')

            # 每 10 个 epoch 保存模型
            if (epoch + 1) % 10 == 0:
                epoch_dir = os.path.join(checkpoint_dir, f'epoch_{epoch+1}')
                os.makedirs(epoch_dir, exist_ok=True)
                checkpoint_path = os.path.join(epoch_dir, f'deeplabv3plus_segmentation_epoch{epoch+1}.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(f'Saved checkpoint to {checkpoint_path}')

        final_epoch_dir = os.path.join(checkpoint_dir, f'epoch_{num_epochs}')
        os.makedirs(final_epoch_dir, exist_ok=True)
        final_checkpoint_path = os.path.join(final_epoch_dir, f'deeplabv3plus_segmentation_epoch{num_epochs}.pth')
        torch.save(model.state_dict(), final_checkpoint_path)
        print(f'Saved final model to {final_checkpoint_path}')

    # 开始训练
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100)