import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from torch.amp import GradScaler, autocast

# 自定义数据集类
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
        
        # 检查掩码值
        mask_np = mask.numpy()
        unique_values = np.unique(mask_np)
        if not np.all(np.isin(unique_values, [0, 1])):
            print(f"Warning: Mask {mask_path} contains invalid values: {unique_values}")
            if mask_np.max() > 1:
                mask_np = (mask_np / 255).astype(np.uint8)
                mask = torch.from_numpy(mask_np)
        
        return image, mask

# 定义极轻量分割模型
class UltraLightSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(UltraLightSegmentation, self).__init__()
        print("Initializing UltraLightSegmentation model")
        # 使用 MobileNetV3-Small 作为骨干网络
        self.backbone = mobilenet_v3_small(weights='IMAGENET1K_V1')
        in_features = 576  # MobileNetV3-Small 的输出通道数
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_features, 32, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.Upsample(size=(640, 512), mode='bilinear', align_corners=False),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        # 提取 MobileNetV3-Small 的特征
        x = self.backbone.features(x)  # 输出形状约为 [batch_size, 576, 20, 16]
        x = self.upsample(x)  # 输出形状为 [batch_size, num_classes, 640, 512]
        return x

# 数据预处理
image_transform = transforms.Compose([
    transforms.Resize((640, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((640, 512), interpolation=Image.NEAREST),
    transforms.ToTensor()
])

# 数据集路径
train_image_dir = 'E:/Robotics/Work/cv/dataset/train/images'
train_mask_dir = 'E:/Robotics/Work/cv/dataset/train/masks'
val_image_dir = 'E:/Robotics/Work/cv/dataset/test/images'
val_mask_dir = 'E:/Robotics/Work/cv/dataset/test/masks'

# 加载数据集
train_dataset = SegmentationDataset(train_image_dir, train_mask_dir, transform=image_transform, mask_transform=mask_transform)
val_dataset = SegmentationDataset(val_image_dir, val_mask_dir, transform=image_transform, mask_transform=mask_transform)

# 设置 DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)

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
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    # 验证参数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {count_parameters(model)}")

    # 训练循环
    def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=35):
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

    # 开始训练
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=35)
    
    # 保存模型
    torch.save(model.state_dict(), 'ultra_light_segmentation.pth')