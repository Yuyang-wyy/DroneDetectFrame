import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# 定义基于EfficientNet的分割模型（与训练代码一致）
class EfficientNetSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetSegmentation, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        in_features = self.backbone._conv_head.out_channels  # 1280 for B0
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_features, 512, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.Upsample(size=(512, 640), mode='bilinear', align_corners=False),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        x = self.backbone.extract_features(x)
        x = self.upsample(x)
        return x

# 自定义测试数据集类
class TestDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir)) if mask_dir else None
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            raise e
        
        if self.transform:
            image = self.transform(image)
        
        # 如果有掩码，加载并处理
        mask = None
        if self.masks:
            mask_path = os.path.join(self.mask_dir, self.masks[idx])
            try:
                mask = Image.open(mask_path).convert('L')
                if self.mask_transform:
                    mask = self.mask_transform(mask)
                else:
                    mask = transforms.ToTensor()(mask)
                # 检查掩码值
                mask_np = mask.numpy()
                unique_values = np.unique(mask_np)
                if not np.all(np.isin(unique_values, [0, 1])):
                    print(f"Warning: Mask {mask_path} contains invalid values: {unique_values}")
            except Exception as e:
                print(f"Error loading {mask_path}: {e}")
                raise e
        
        return image, mask, self.images[idx]

# 计算mIoU
def compute_miou(preds, targets, num_classes):
    iou_per_class = []
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        if union == 0:
            iou_per_class.append(np.nan)
        else:
            iou_per_class.append(intersection / union)
    return np.nanmean(iou_per_class)

# 测试函数
def test_model(model, test_loader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    miou_scores = []
    
    with torch.no_grad():
        for images, masks, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  # 获取预测类别
            
            # 保存预测掩码
            for i, pred in enumerate(preds):
                pred_np = pred.cpu().numpy().astype(np.uint8) * 255  # 转换为0, 255以便可视化
                pred_img = Image.fromarray(pred_np)
                pred_img.save(os.path.join(output_dir, f"pred_{filenames[i]}.png"))
                
                # 计算mIoU（如果有真实掩码）
                if masks is not None:
                    miou = compute_miou(pred, masks[i].squeeze(0), num_classes=2)
                    miou_scores.append(miou)
                    print(f"Image {filenames[i]}, mIoU: {miou:.4f}")
    
    if miou_scores:
        mean_miou = np.mean(miou_scores)
        print(f"Mean mIoU: {mean_miou:.4f}")

# 主程序
if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # 数据预处理
    image_transform = transforms.Compose([
        transforms.Resize((512, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((512, 640), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])
    
    # 测试数据集路径（请根据你的本地路径修改）
    test_image_dir = 'E:/Robotics/Work/cv/dataset/test/images'
    test_mask_dir = 'E:/Robotics/Work/cv/dataset/test/masks'  # 如果没有真实掩码，设为None
    output_dir = 'E:/Robotics/Work/cv/results'
    
    # 加载测试数据集
    test_dataset = TestDataset(test_image_dir, test_mask_dir, transform=image_transform, mask_transform=mask_transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
    # 加载模型
    model = EfficientNetSegmentation(num_classes=2).to(device)
    model.load_state_dict(torch.load('E:/Robotics/Work/cv/codes/model/efficientnet_segmentation_epoch{num_epochs}.pth.pth', map_location=device))
    
    # 测试模型
    test_model(model, test_loader, device, output_dir)
    print(f"Predictions saved in {output_dir}")