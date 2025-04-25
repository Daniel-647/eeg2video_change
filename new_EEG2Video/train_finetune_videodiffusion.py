# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import os.path as osp
import os
import sys
import warnings
import torch
import numpy as np

from wan import WanT2V
from wan.configs import WAN_CONFIGS
from torch.utils.data import DataLoader, Dataset

class EEGVideoDataset(Dataset):
    def __init__(self, eeg_features, video_features):
        self.eeg_features = eeg_features
        self.video_features = video_features
    
    def __len__(self):
        return len(self.eeg_features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.eeg_features[idx]), torch.FloatTensor(self.video_features[idx])

def train_model(args):
    # 初始化模型
    cfg = WAN_CONFIGS['t2v-1.3B']
    model = WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
    )
    
    # 加载EEG和视频特征数据
    eeg_features = np.load(args.eeg_feature_path)
    video_features = np.load(args.video_feature_path)
    
    # 创建数据加载器
    dataset = EEGVideoDataset(eeg_features, video_features)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 训练循环
    for epoch in range(args.num_epochs):
        for batch_idx, (eeg_batch, video_batch) in enumerate(dataloader):
            # 将EEG特征作为条件输入模型
            loss = model.train_step(eeg_batch, video_batch)
            
            if batch_idx % args.log_interval == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item()}')
    
    # 保存微调后的模型
    torch.save(model.state_dict(), args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune Wan2.1 model with EEG features')
    parser.add_argument('--eeg_feature_path', type=str, required=True, help='Path to EEG features numpy file')
    parser.add_argument('--video_feature_path', type=str, required=True, help='Path to video features numpy file')
    parser.add_argument('--ckpt_dir', type=str, default='cache', help='Path to checkpoint directory')
    parser.add_argument('--device_id', type=int, default=0, help='GPU device ID')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save_path', type=str, default='eeg2video_model.pth', help='Path to save trained model')
    
    args = parser.parse_args()
    train_model(args)