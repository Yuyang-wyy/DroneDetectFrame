import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

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

# 主程序
if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    model = UltraLightSegmentation(num_classes=2).to(device)
    pth_path = 'E:/Robotics/Work/cv/codes/deepmodel/epoch_40/deeplabv3plus_segmentation_epoch40.pth'
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()

    # 创建虚拟输入
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 256, 320).to(device)

    # ONNX 导出路径
    onnx_path = 'E:/Robotics/Work/cv/ultralight_segmentation.onnx'

    # 导出 ONNX 模型
    print("Exporting model to ONNX...")
    torch.onnx.export(
        model,
        input_tensor,
        onnx_path,
        export_params=True,
        opset_version=11,  # 支持常见操作，兼容大多数推理引擎
        do_constant_folding=True,  # 优化常量折叠
        input_names=['input'],  # 输入名称
        output_names=['output'],  # 输出名称
        dynamic_axes={
            'input': {0: 'batch_size'},  # 支持动态 batch_size
            'output': {0: 'batch_size'}  # 输出动态 batch_size
        }
    )
    print(f"ONNX model saved to {onnx_path}")

    # 验证 ONNX 模型
    print("Validating ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed.")

    # 可选：测试 ONNX 推理
    print("Testing ONNX inference...")
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    dummy_input = input_tensor.cpu().numpy()
    outputs = ort_session.run(None, {input_name: dummy_input})[0]
    print(f"ONNX inference output shape: {outputs.shape}")

    # 可选：验证 PyTorch 和 ONNX 输出一致性
    with torch.no_grad():
        pytorch_output = model(input_tensor).cpu().numpy()
    np.testing.assert_allclose(outputs, pytorch_output, rtol=1e-2, atol=1e-3, err_msg="ONNX and PyTorch outputs differ.")
    print("ONNX and PyTorch outputs are consistent.")