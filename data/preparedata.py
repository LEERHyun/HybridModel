import lmdb
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import os
from dataset import SIDD_Medium, DenoisingDataset, PolyUDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch

# -*- coding: utf-8 -*-

def save_dataset_to_lmdb(dataset, save_path, dataset_name="dataset", map_size_gb = 13):
    """
    PyTorch Dataset을 LMDB로 저장
    
    Args:
        dataset: PyTorch Dataset 또는 Subset
        save_path: LMDB를 저장할 경로
        dataset_name: 데이터셋 이름 (train/val/test)
    """
    os.makedirs(save_path, exist_ok=True)
    
    # LMDB 환경 생성 (map_size는 충분히 크게 설정)
    env = lmdb.open(save_path, map_size=map_size_gb*1024**3)  # 1TB
    
    print(f"{dataset_name} 데이터셋을 LMDB로 저장 중...")
    
    with env.begin(write=True) as txn:
        for idx in tqdm(range(len(dataset)), desc=f"Saving {dataset_name}"):
            # 데이터 로드 (noisy, gt)
            noisy_img, gt_img = dataset[idx]
            
            # Tensor를 numpy로 변환 (메모리 효율)
            if torch.is_tensor(noisy_img):
                noisy_img = noisy_img.numpy()
            if torch.is_tensor(gt_img):
                gt_img = gt_img.numpy()
            
            # 데이터 딕셔너리 생성
            data = {
                'noisy': noisy_img,
                'gt': gt_img
            }
            
            # Pickle로 직렬화
            data_byte = pickle.dumps(data)
            
            # 키-값 쌍으로 저장
            key = f'{idx:08d}'.encode('ascii')
            txn.put(key, data_byte)
        
        # 메타데이터 저장 (데이터셋 크기)
        txn.put(b'__len__', str(len(dataset)).encode('ascii'))
    
    env.close()
    print(f'{dataset_name} LMDB 저장 완료: {save_path}')
    print(f'총 {len(dataset)}개의 이미지 쌍 저장됨\n')

def create_lmdb_from_dataset(root_dataset_dir, output_lmdb_dir, split_ratios=(0.8, 0.1, 0.1)):
    """
    기존 Dataset을 Train/Val/Test로 분할하여 각각 LMDB로 저장
    
    Args:
        root_dataset_dir: 원본 데이터셋 경로
        output_lmdb_dir: LMDB 저장 경로
        split_ratios: (train, val, test) 비율 튜플
    """
    # 데이터셋 로드
    print("원본 데이터셋 로딩 중...")
    dataset = SIDD_Medium(root_dir=root_dataset_dir, transform=None)
    data_size = len(dataset)
    
    # 분할 크기 계산
    train_ratio, val_ratio, test_ratio = split_ratios
    train_size = int(train_ratio * data_size)
    val_size = int(val_ratio * data_size)
    test_size = data_size - train_size - val_size
    
    print(f"전체 데이터셋 크기: {data_size}")
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}\n")
    
    # 데이터셋 분할
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 재현성을 위한 시드 설정
    )
    
    # 각 분할을 LMDB로 저장
    save_dataset_to_lmdb(
        train_dataset, 
        os.path.join(output_lmdb_dir, 'train_lmdb'),
        "Train",
        map_size_gb= 15
    )
    
    save_dataset_to_lmdb(
        val_dataset, 
        os.path.join(output_lmdb_dir, 'val_lmdb'),
        "Validation",
        map_size_gb= 2
    )
    
    save_dataset_to_lmdb(
        test_dataset, 
        os.path.join(output_lmdb_dir, 'test_lmdb'),
        "Test",
        map_size_gb= 2
    )
    
    print("=" * 50)
    print("모든 데이터셋이 LMDB로 저장 완료!")
    print(f"저장 경로: {output_lmdb_dir}")

if __name__ == "__main__":
    dataset_dir = r"D:\Dataset\Denoising\Whole_Dataset_256\SIDD_patched_256"
    output_dir = r"C:\Users\Ahhyun\Desktop\Workplace\Code\Denoising_1231"

    dataset = SIDD_Medium(root_dir=dataset_dir, transform=None)
    data_size = len(dataset)
    train_size = int(0.8 * data_size)
    val_size = int(0.1 * data_size)
    test_size = data_size - train_size - val_size
    
    create_lmdb_from_dataset(dataset_dir,output_lmdb_dir=output_dir)
