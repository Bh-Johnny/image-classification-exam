import click
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import TarImageDataset
from predictor import BatchClassifier
import os

@click.command()
@click.option('--tar_path', default='./data/zip1.tar', help='Path to the tar file (e.g., data/zip1.tar)')
@click.option('--index_path', default='./data/index.parquet', help='Path to the parquet index (e.g., data/index.parquet)')
@click.option('--output_path', default='results.csv', help='Path to save results')
@click.option('--batch_size', default=32, help='Batch size for inference')
@click.option('--num_workers', default=0, help='Number of workers for data loading')
def main(tar_path, index_path, output_path, batch_size, num_workers):

    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"Tar file not found: {tar_path}")
    
    # 1. 初始化模型
    classifier = BatchClassifier()
    
    # 2. 初始化数据集和 DataLoader
    dataset = TarImageDataset(
        tar_path=tar_path, 
        parquet_path=index_path, 
        transform=classifier.get_transform()
    )
    pin_memory = torch.backends.cuda.is_built()  # 只有 CUDA 支持 pin_memory
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory # 动态设置
    )

    results = []
    
    print(f"Starting inference on {len(dataset)} images with batch_size={batch_size}...")
    
    # 3. Batch 推理循环
    for batch in tqdm(loader):
        pixel_values = batch['pixel_values']
        keys = batch['key']
        valids = batch['valid']
        
        # 过滤掉读取失败的坏图 (valid=False)
        valid_mask = valids == True
        if not valid_mask.any():
            continue
            
        valid_pixels = pixel_values[valid_mask]
        valid_keys = [k for k, v in zip(keys, valids) if v]
        
        # 执行推理
        indices, probs = classifier.predict_batch(valid_pixels)
        
        # 收集结果
        for key, idx, prob in zip(valid_keys, indices, probs):
            results.append({
                "key": key,
                "label": classifier.class_names[idx], # "anime" or "realistic"
                "score": float(prob)
            })

    # 4. 保存结果
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    
    print(f"Done! Results saved to {output_path}")
    
    # 打印简单统计
    print("Statistics:")
    print(df_results['label'].value_counts())

if __name__ == '__main__':
    main()