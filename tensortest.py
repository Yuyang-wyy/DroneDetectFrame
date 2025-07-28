import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
np.bool = np.bool_
import os
from PIL import Image
import torch
from torchvision import transforms
from scipy.special import softmax
import matplotlib.pyplot as plt

class TensorRTInference:
    def __init__(self, engine_path, verbose=False):
        """初始化TensorRT推理器"""
        print(f"🔧 初始化TensorRT推理器...")
        print(f"引擎路径: {engine_path}")
        
        self.verbose = verbose
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        
        # 加载引擎
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        
        # 获取输入输出信息
        self.get_engine_info()
        
        # 分配GPU内存
        self.allocate_buffers()
        
        # 数据预处理 - 与PyTorch一致
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ TensorRT推理器初始化完成")
    
    def load_engine(self, engine_path):
        """加载TensorRT引擎"""
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"引擎文件不存在: {engine_path}")
        
        print("📂 加载TensorRT引擎...")
        runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine is None:
            raise RuntimeError("引擎加载失败")
        
        print("✅ TensorRT引擎加载成功")
        return engine
    
    def get_engine_info(self):
        """获取引擎输入输出信息"""
        self.input_name = None
        self.output_name = None
        self.input_dtype = None
        self.output_dtype = None
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            mode = self.engine.get_tensor_mode(name)
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_dtype = trt.nptype(dtype)
            else:
                self.output_name = name
                self.output_dtype = trt.nptype(dtype)
    
    def allocate_buffers(self, max_batch_size=1):
        """分配GPU内存缓冲区"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                size = max_batch_size * 3 * 256 * 320
                device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
                self.inputs.append({'name': name, 'device': device_mem, 'dtype': dtype})
            else:
                size = max_batch_size * 2 * 512 * 640  # 2 classes, output resolution
                device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
                host_mem = cuda.pagelocked_empty(size, dtype)
                self.outputs.append({'name': name, 'host': host_mem, 'device': device_mem, 'dtype': dtype})
            
            self.bindings.append(int(device_mem))
    
    def preprocess_image(self, image_path):
        """预处理图像"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_transform(image)
        return image_tensor.unsqueeze(0).numpy()
    
    def generate_heatmap(self, input_dir, output_dir):
        """生成热力图"""
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取输入图像
        image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"Found {len(image_files)} images in {input_dir}")
        
        # 预热引擎
        print("🔥 预热TensorRT引擎...")
        dummy_input = np.random.randn(1, 3, 256, 320).astype(self.input_dtype)
        self.context.set_input_shape(self.input_name, (1, 3, 256, 320))
        for _ in range(10):
            cuda.memcpy_htod(self.inputs[0]['device'], dummy_input)
            self.context.execute_v2(self.bindings)
            cuda.memcpy_dtoh(self.outputs[0]['host'], self.outputs[0]['device'])
        print("✅ 预热完成")
        
        # 逐张图像推理
        for image_file in image_files:
            # 加载和预处理图像
            img_path = os.path.join(input_dir, image_file)
            try:
                image_tensor = self.preprocess_image(img_path)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
            
            # 设置输入形状
            self.context.set_input_shape(self.input_name, (1, 3, 256, 320))
            
            # 复制输入数据到GPU
            cuda.memcpy_htod(self.inputs[0]['device'], image_tensor)
            
            # 推理
            success = self.context.execute_v2(self.bindings)
            if not success:
                print(f"❌ TensorRT推理失败 for {image_file}")
                continue
            
            # 复制输出数据到CPU
            cuda.memcpy_dtoh(self.outputs[0]['host'], self.outputs[0]['device'])
            
            # 后处理
            output_shape = self.context.get_tensor_shape(self.output_name)
            batch_output = self.outputs[0]['host'][:np.prod(output_shape)].reshape(output_shape)
            batch_probs = softmax(batch_output, axis=1)  # 形状为 (1, num_classes, height, width)
            foreground_probs = batch_probs[0, 1, :, :]  # 前景类概率，形状 (height, width)
            
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
    
    def __del__(self):
        """清理资源"""
        try:
            for inp in self.inputs:
                if hasattr(inp['device'], 'free'):
                    inp['device'].free()
            for out in self.outputs:
                if hasattr(out['device'], 'free'):
                    out['device'].free()
        except:
            pass

# 主程序
if __name__ == '__main__':
    # 配置参数
    config = {
        'engine_path': './ultralight_segmentation_256x320_fp16.trt',
        'input_dir': 'E:/Robotics/Work/cv/newpics',
        'output_dir': 'E:/Robotics/Work/cv/deep_final_results/tensorheatmap',
    }
    
    print("=" * 60)
    print("🚀 TensorRT DeepLabv3+ 热力图生成")
    print("=" * 60)
    
    # 初始化推理器
    try:
        inferencer = TensorRTInference(config['engine_path'], verbose=True)
    except Exception as e:
        print(f"❌ TensorRT推理器初始化失败: {e}")
        exit()
    
    # 生成热力图
    inferencer.generate_heatmap(config['input_dir'], config['output_dir'])
    print(f"Heatmaps saved in {config['output_dir']}")