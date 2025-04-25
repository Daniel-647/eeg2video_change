# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import os
import numpy as np
import torch

from wan import WanT2V
from wan.configs import WAN_CONFIGS

class EEGVideoGenerator:
    def __init__(self, model_path, device_id=0):
        # 初始化模型
        cfg = WAN_CONFIGS['t2v-1.3B']
        self.model = WanT2V(
            config=cfg,
            checkpoint_dir=None,
            device_id=device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
        )
        
        # 加载训练好的权重
        state_dict = torch.load(model_path, map_location=f'cuda:{device_id}')
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def generate_video(self, eeg_feature, size=(480, 832), sampling_steps=50, guide_scale=6.0, shift_scale=8.0):
        """
        根据EEG特征生成视频
        
        参数:
            eeg_feature: EEG特征向量
            size: 生成视频的分辨率 (宽, 高)
            sampling_steps: 扩散步数
            guide_scale: 引导尺度
            shift_scale: 位移尺度
        """
        with torch.no_grad():
            # 将EEG特征转换为模型输入格式
            eeg_tensor = torch.FloatTensor(eeg_feature).unsqueeze(0).to(f'cuda:{self.model.device_id}')
            
            # 生成视频
            video = self.model.generate(
                condition=eeg_tensor,
                size=size,
                shift=shift_scale,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                offload_model=True
            )
            
            return video

def main():
    parser = argparse.ArgumentParser(description='Generate video from EEG features using Wan2.1 model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--eeg_feature', type=str, required=True, help='EEG feature numpy file path')
    parser.add_argument('--output_path', type=str, default='generated_video.mp4', help='Output video path')
    parser.add_argument('--device_id', type=int, default=0, help='GPU device ID')
    parser.add_argument('--width', type=int, default=480, help='Video width')
    parser.add_argument('--height', type=int, default=832, help='Video height')
    parser.add_argument('--sampling_steps', type=int, default=50, help='Diffusion sampling steps')
    parser.add_argument('--guide_scale', type=float, default=6.0, help='Guidance scale')
    parser.add_argument('--shift_scale', type=float, default=8.0, help='Shift scale')
    
    args = parser.parse_args()
    
    # 初始化生成器
    generator = EEGVideoGenerator(args.model_path, args.device_id)
    
    # 加载EEG特征
    eeg_feature = np.load(args.eeg_feature)
    
    # 生成视频
    video = generator.generate_video(
        eeg_feature,
        size=(args.width, args.height),
        sampling_steps=args.sampling_steps,
        guide_scale=args.guidance_scale,
        shift_scale=args.shift_scale
    )
    
    # 保存视频
    from wan.utils.utils import cache_video
    cache_video(
        tensor=video[None],
        save_file=args.output_path,
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )
    
    print(f'Video saved to {args.output_path}')

if __name__ == '__main__':
    main()