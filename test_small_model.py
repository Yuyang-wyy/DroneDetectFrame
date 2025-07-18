import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# 定义与训练一致的 UltraLightSegmentation 模型
class UltraLightSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(UltraLightSegmentation, self).__init__()
        print("Initializing UltraLightSegmentation model with MobileNetV3-Small")
        self.backbone = mobilenet_v3_small(weights='IMAGENET1K_V1')
        in_features = 576  # MobileNetV3-Small 的输出通道数
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_features, 128, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.Upsample(size=(512, 640), mode='bilinear', align_corners=False),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        x = self.backbone.features(x)  # 输出形状约为 [batch_size, 576, 20, 16]
        x = self.upsample(x)  # 输出形状为 [batch_size, num_classes, 640, 512]
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

# 计算 mIoU
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
    
    # 推理时间测量
    inference_times = []
    total_samples = 0
    
    with torch.no_grad():
        for images, masks, filenames in test_loader:
            images = images.to(device)
            total_samples += images.size(0)
            
            # 测量推理时间
            if device.type == 'cuda':
                torch.cuda.synchronize()
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                outputs = model(images)
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time) / 1000.0  # 转换为秒
            else:
                start_time = time.perf_counter()
                outputs = model(images)
                inference_time = time.perf_counter() - start_time
            
            inference_times.append(inference_time)
            
            preds = torch.argmax(outputs, dim=1)
            
            # 保存预测掩码
            for i, pred in enumerate(preds):
                pred_np = pred.cpu().numpy().astype(np.uint8) * 255
                pred_img = Image.fromarray(pred_np)
                pred_img.save(os.path.join(output_dir, f"pred_{filenames[i]}.png"))
                
                if masks is not None:
                    miou = compute_miou(pred, masks[i].squeeze(0), num_classes=2)
                    miou_scores.append(miou)
                    print(f"Image {filenames[i]}, mIoU: {miou:.4f}")
    
    if miou_scores:
        mean_miou = np.mean(miou_scores)
        print(f"Mean mIoU: {mean_miou:.4f}")
    
    # 打印推理时间统计
    avg_inference_time = np.mean(inference_times)
    total_inference_time = np.sum(inference_times)
    print(f"Average inference time per batch: {avg_inference_time:.4f} seconds")
    print(f"Total inference time for {total_samples} samples: {total_inference_time:.4f} seconds")
    print(f"Average inference time per sample: {total_inference_time / total_samples:.6f} seconds")

# 主程序
if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # 数据预处理（与训练一致）
    image_transform = transforms.Compose([
        transforms.Resize((512, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((512, 640), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])
    
    # 测试数据集路径
    test_image_dir = 'E:/Robotics/Work/cv/dataset/test/images'
    test_mask_dir = 'E:/Robotics/Work/cv/dataset/test/masks'  # 如果没有真实掩码，设为 None
    output_dir = 'E:/Robotics/Work/cv/small_results'
    
    # 加载测试数据集
    test_dataset = TestDataset(test_image_dir, test_mask_dir, transform=image_transform, mask_transform=mask_transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
    # 加载模型
    model = UltraLightSegmentation(num_classes=2).to(device)
    model.load_state_dict(torch.load('E:/Robotics/Work/cv/codes/smallmodel/epoch_400/efficientnet_segmentation_epoch400.pth', map_location=device))
    
    # 测试模型
    test_model(model, test_loader, device, output_dir)
    print(f"Predictions saved in {output_dir}")