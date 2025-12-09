import io
import pandas as pd
from PIL import Image
import torch
import os
from torch.utils.data import Dataset

class TarImageDataset(Dataset):
    def __init__(self, tar_path, parquet_path, transform=None):
        self.tar_path = tar_path
        self.transform = transform
        
        # 1. 读取索引文件
        print(f"Loading index from {parquet_path}...")
        full_df = pd.read_parquet(parquet_path)
        
        # 2. 取出当前 tar 文件
        filename_with_ext = os.path.basename(tar_path)
        # 去掉后缀（变成 'zip1'），以匹配 parquet 中的格式
        tar_name_stem = filename_with_ext.replace('.tar', '')
        self.meta_data = full_df[
            (full_df['tar_name'] == tar_name_stem) | 
            (full_df['tar_name'] == filename_with_ext)
        ].reset_index(drop=True)

        if len(self.meta_data) == 0:
            raise ValueError(f"No images found for {tar_path}. Check if 'tar_name' in parquet matches '{tar_name_stem}'")
        
        # 文件句柄初始化为 None (Lazy loading)
        self.tar_handle = None

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        # 确保每个 Worker 都有独立的文件句柄
        if self.tar_handle is None:
            self.tar_handle = open(self.tar_path, "rb")

        row = self.meta_data.iloc[idx]
        offset = row['image_offset']
        size = row['image_size']
        key = row['key']

        try:
            # 移动指针并读取
            self.tar_handle.seek(offset)
            image_bytes = self.tar_handle.read(size)
            
            # 解码图片
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # 应用预处理 (CLIP Processor)
            if self.transform:
                # transform 返回的是一个 dict {'pixel_values': tensor}
                image_tensor = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image_tensor = image
                
            return {
                "pixel_values": image_tensor,
                "key": key,
                "valid": True
            }
            
        except Exception as e:
            # 遇到坏图时标记
            print(f"Error reading image {key}: {e}")
            # 返回一个全零的 dummy tensor 保持 batch 形状一致
            return {
                "pixel_values": torch.zeros((3, 224, 224)), # 假设 CLIP 输入是 224
                "key": key,
                "valid": False
            }