import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 创建伪彩色掩码（用于可视化）
def create_colored_mask(mask_np):
    # mask_np: [H, W]，值 0 或 1（或 0 或 255）
    mask_np = mask_np / 255 if mask_np.max() > 1 else mask_np  # 归一化到 0-1
    colored_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    colored_mask[mask_np >= 0.5] = [255, 0, 0]  # 前景（红色）
    colored_mask[mask_np < 0.5] = [0, 0, 0]     # 背景（黑色）
    return colored_mask

# 主程序
if __name__ == '__main__':
    # 文件夹路径
    image_dir = 'E:/Robotics/Work/cv/dataset/test/images'
    mask_dir = 'E:/Robotics/Work/cv/dataset/test/masks'
    pred_dir = 'E:/Robotics/Work/cv/deep_smaller_results'
    output_dir = 'E:/Robotics/Work/cv/codes/compare_results_smaller'
    num_samples = 10  # 随机抽取的样本数量

    # 获取文件名
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    pred_files = [f for f in os.listdir(pred_dir) if f.startswith('pred_') and (f.endswith('.png') or f.endswith('.png.png'))]
    
    # 找到匹配的三元组
    valid_images = []
    for img_name in image_files:
        base_name = os.path.splitext(img_name)[0]  # 去掉扩展名，例如 'image_21'
        pred_name = f"pred_{base_name}.png"
        pred_name_double = f"pred_{base_name}.png.png"
        if (pred_name in pred_files or pred_name_double in pred_files) and img_name in mask_files:
            valid_images.append(img_name)
    
    if not valid_images:
        raise ValueError("No matching image, mask, and prediction pairs found. Check file names and directories.")
    
    # 随机抽取样本
    if len(valid_images) < num_samples:
        print(f"Warning: Only {len(valid_images)} matching pairs found, using all.")
        num_samples = len(valid_images)
    selected_images = random.sample(valid_images, num_samples)
    
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    # 拼接并保存图像
    for img_name in selected_images:
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)
        # 优先尝试双重后缀文件名
        pred_path = os.path.join(pred_dir, f"pred_{base_name}.png.png")
        if not os.path.exists(pred_path):
            pred_path = os.path.join(pred_dir, f"pred_{base_name}.png")
        
        try:
            # 加载图像
            image = np.array(Image.open(img_path).convert('RGB'))  # [H, W, 3]
            mask = np.array(Image.open(mask_path).convert('L'))    # [H, W]
            pred = np.array(Image.open(pred_path).convert('L'))    # [H, W]
        except Exception as e:
            print(f"Error loading {img_name} (image: {img_path}, mask: {mask_path}, pred: {pred_path}): {e}")
            continue
        
        # 转换为伪彩色掩码
        mask_colored = create_colored_mask(mask)
        pred_colored = create_colored_mask(pred)
        
        # 使用 Matplotlib 拼接
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 调整画布大小
        axes[0].imshow(image)
        axes[0].set_title('Original')
        axes[0].axis('off')
        axes[1].imshow(mask_colored)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        axes[2].imshow(pred_colored)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(output_dir, f"compare_{img_name}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved comparison image to {output_path}")
    
    print(f"All {len(selected_images)} comparison images saved in {output_dir}")