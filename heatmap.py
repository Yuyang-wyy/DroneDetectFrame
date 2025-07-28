import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

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

# 定义 UltraLightSegmentation 模型
class UltraLightSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(UltraLightSegmentation, self).__init__()
        print("Initializing UltraLightSegmentation model with MobileNetV3-Small and DeepLabv3+ Decoder")
        self.backbone = mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.aspp = ASPP(in_channels=576, out_channels=128)
        self.decoder = DeepLabDecoder(low_level_channels=16, num_classes=num_classes)

    def forward(self, x):
        input_size = x.size()[2:]
        low_level_feat = self.backbone.features[0](x)
        x = self.backbone.features(x)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x

# 生成热力图的函数
def generate_heatmap(model, input_dir, output_dir, device):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据预处理
    image_transform = transforms.Compose([
        transforms.Resize((256, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 获取输入图像
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {len(image_files)} images in {input_dir}")
    
    model.eval()
    with torch.no_grad():
        for image_file in image_files:
            # 加载和预处理图像
            img_path = os.path.join(input_dir, image_file)
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
            
            image_tensor = image_transform(image).unsqueeze(0).to(device)
            
            # 模型推理
            outputs = model(image_tensor)
            
            # 转换为概率
            probs = torch.softmax(outputs, dim=1)
            foreground_probs = probs[0, 1, :, :].cpu().numpy()  # 前景类概率，形状 (height, width)
            
            # 生成并保存热力图
            plt.figure(figsize=(8, 6))
            plt.imshow(foreground_probs, cmap='hot', vmin=0, vmax=1)
            plt.colorbar(label='Foreground Probability')
            plt.title(f'Heatmap: {image_file}')
            plt.axis('off')
            
            # 保存热力图
            heatmap_path = os.path.join(output_dir, f"heatmap_{image_file}")
            plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            print(f"Saved heatmap for {image_file} at {heatmap_path}")

# 主程序
if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # 模型和权重路径
    model = UltraLightSegmentation(num_classes=2).to(device)
    weights_path = 'E:/Robotics/Work/cv/codes/deepmodel_final/epoch_50/deeplabv3plus_segmentation_epoch50.pth'
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    # 输入输出路径
    input_dir = 'E:/Robotics/Work/cv/newpics'
    output_dir = 'E:/Robotics/Work/cv/deep_final_results/heatmap'
    
    # 生成热力图
    generate_heatmap(model, input_dir, output_dir, device)
    print(f"Heatmaps saved in {output_dir}")