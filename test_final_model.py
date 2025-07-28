import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# 定义 DeepLabv3+ 的 ASPP 模块
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=128, atrous_rates=[6, 12, 18]):
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
    def __init__(self, low_level_channels, num_classes, aspp_out_channels=128):
        super(DeepLabDecoder, self).__init__()
        self.low_level_conv = nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False)
        self.low_level_bn = nn.BatchNorm2d(48)
        self.low_level_relu = nn.ReLU()
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(aspp_out_channels + 48, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x, low_level_feat):
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=False)
        low_level_feat = self.low_level_conv(low_level_feat)
        low_level_feat = self.low_level_bn(low_level_feat)
        low_level_feat = self.low_level_relu(low_level_feat)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.decoder_conv(x)
        return x

# 定义 UltraLightSegmentation 模型（使用 DeepLabv3+ 解码器）
class UltraLightSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(UltraLightSegmentation, self).__init__()
        print("Initializing UltraLightSegmentation model with MobileNetV3-Small and DeepLabv3+ Decoder")
        self.backbone = mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.aspp = ASPP(in_channels=576, out_channels=128)  # MobileNetV3-Small 输出通道数为 576
        self.decoder = DeepLabDecoder(low_level_channels=16, num_classes=num_classes)  # 低层特征来自 conv1

    def forward(self, x, timing=False):
        input_size = x.size()[2:]  # Save input resolution (512, 640)
        
        # Initialize timing variables
        backbone_time = 0.0
        decoder_time = 0.0
        
        # Backbone timing
        if timing and torch.cuda.is_available():
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
        elif timing:
            start_time = time.perf_counter()
        
        low_level_feat = self.backbone.features[0](x)  # Extract low-level features
        x = self.backbone.features(x)  # Extract high-level features
        
        if timing and torch.cuda.is_available():
            end_time.record()
            torch.cuda.synchronize()
            backbone_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        elif timing:
            backbone_time = time.perf_counter() - start_time
        
        # Decoder timing (ASPP + DeepLabDecoder)
        if timing and torch.cuda.is_available():
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
        elif timing:
            start_time = time.perf_counter()
        
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        if timing and torch.cuda.is_available():
            end_time.record()
            torch.cuda.synchronize()
            decoder_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        elif timing:
            decoder_time = time.perf_counter() - start_time
        
        if timing:
            return x, backbone_time, decoder_time
        return x

# 自定义测试数据集类（保持不变）
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
        
        mask = None
        if self.masks:
            mask_path = os.path.join(self.mask_dir, self.masks[idx])
            try:
                mask = Image.open(mask_path).convert('L')
                if self.mask_transform:
                    mask = self.mask_transform(mask)
                else:
                    mask = transforms.ToTensor()(mask)
                mask_np = mask.numpy()
                unique_values = np.unique(mask_np)
                if not np.all(np.isin(unique_values, [0, 1])):
                    print(f"Warning: Mask {mask_path} contains invalid values: {unique_values}")
            except Exception as e:
                print(f"Error loading {mask_path}: {e}")
                raise e
        
        return image, mask, self.images[idx]

# 计算 mIoU（保持不变）
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

# 测试函数（修改为输出概率值）
def test_model(model, test_loader, device, output_dir, threshold=0.5):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    miou_scores = []
    
    # 推理时间测量
    inference_times = []
    backbone_times = []
    decoder_times = []
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
                outputs, backbone_time, decoder_time = model(images, timing=True)
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            else:
                start_time = time.perf_counter()
                outputs, backbone_time, decoder_time = model(images, timing=True)
                inference_time = time.perf_counter() - start_time
            
            inference_times.append(inference_time)
            backbone_times.append(backbone_time)
            decoder_times.append(decoder_time)
            
            # 将 logits 转换为概率
            probs = torch.softmax(outputs, dim=1)  # 形状为 (batch_size, num_classes, height, width)
            # 提取前景类（类别 1，白色）的概率
            foreground_probs = probs[:, 1, :, :]  # 形状为 (batch_size, height, width)
            
            # 保存概率值
            for i, prob in enumerate(foreground_probs):
                prob_np = prob.cpu().numpy()  # 转换为 numpy 数组，值在 [0, 1]
                prob_path = os.path.join(output_dir, f"prob_{filenames[i]}.npy")
                np.save(prob_path, prob_np)  # 保存为 .npy 文件
                print(f"Saved foreground probability for {filenames[i]} at {prob_path}")
                
                # 如果需要计算 mIoU，基于阈值生成二值掩码
                if masks is not None:
                    preds = (prob > threshold).long()  # 前景类概率 > threshold 则为 1
                    miou = compute_miou(preds, masks[i].squeeze(0), num_classes=2)
                    miou_scores.append(miou)
                    print(f"Image {filenames[i]}, mIoU: {miou:.4f}")
    
    if miou_scores:
        mean_miou = np.mean(miou_scores)
        print(f"Mean mIoU: {mean_miou:.4f}")
    
    # 打印推理时间统计
    avg_inference_time = np.mean(inference_times)
    total_inference_time = np.sum(inference_times)
    avg_backbone_time = np.mean(backbone_times)
    total_backbone_time = np.sum(backbone_times)
    avg_decoder_time = np.mean(decoder_times)
    total_decoder_time = np.sum(decoder_times)
    print(f"Average inference time per batch: {avg_inference_time:.4f} seconds")
    print(f"Total inference time for {total_samples} samples: {total_inference_time:.4f} seconds")
    print(f"Average inference time per sample: {total_inference_time / total_samples:.6f} seconds")
    print(f"Average backbone time per sample: {total_backbone_time / total_samples:.6f} seconds")
    print(f"Average decoder time per sample: {total_decoder_time / total_samples:.6f} seconds")

# 主程序
if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # 数据预处理（与训练一致）
    image_transform = transforms.Compose([
        transforms.Resize((256, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 320), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])
    
    # 测试数据集路径
    test_image_dir = 'E:/Robotics/Work/cv/dataset/test/images'
    test_mask_dir = 'E:/Robotics/Work/cv/dataset/test/masks'
    output_dir = 'E:/Robotics/Work/cv/deep_final_results'
    
    # 加载测试数据集
    test_dataset = TestDataset(test_image_dir, test_mask_dir, transform=image_transform, mask_transform=mask_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    # 加载模型
    model = UltraLightSegmentation(num_classes=2).to(device)
    # 注意：需要使用 DeepLabv3+ 的权重
    model.load_state_dict(torch.load('E:/Robotics/Work/cv/codes/deepmodel_final/epoch_50/deeplabv3plus_segmentation_epoch50.pth', map_location=device))
    
    # Warm-up phase
    print("Starting warm-up phase...")
    model.eval()
    warmup_iterations = 50
    with torch.no_grad():
        # Get a single batch for warm-up (or use dummy input)
        try:
            images, _, _ = next(iter(test_loader))  # Get first batch
            images = images.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            for _ in range(warmup_iterations):
                _ = model(images, timing=False)  # Run forward pass without timing
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            print(f"Warm-up completed ({warmup_iterations} iterations).")
        except Exception as e:
            print(f"Error during warm-up: {e}")
            print("Falling back to dummy input for warm-up...")
            dummy_input = torch.randn(1, 3, 256, 320).to(device)  # Match batch_size=1 and input size
            if device.type == 'cuda':
                torch.cuda.synchronize()
            for _ in range(warmup_iterations):
                _ = model(dummy_input, timing=False)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            print(f"Warm-up with dummy input completed ({warmup_iterations} iterations).")
    
    # 测试模型
    test_model(model, test_loader, device, output_dir, threshold=0.5)
    print(f"Probability maps saved in {output_dir}")