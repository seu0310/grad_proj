import sys, types

# --- 🔧 torchvision의 lzma import 우회용 더미 모듈 생성 ---
if 'lzma' not in sys.modules:
    fake_lzma = types.SimpleNamespace()
    fake_lzma.open = lambda *args, **kwargs: None  # 더미 open 함수 추가
    sys.modules['lzma'] = fake_lzma
# -----------------------------------------------------------

import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch.multiprocessing
import glob
torch.multiprocessing.set_sharing_strategy('file_system')


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, attr_path, partition_path, partition_type, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        self.partition_df = pd.read_csv(partition_path, delim_whitespace=True, header=None)
        self.partition_df.columns = ['image_id', 'partition']
        partition_map = {'train': 0, 'val': 1, 'test': 2}
        self.partition_df = self.partition_df[self.partition_df['partition'] == partition_map[partition_type]]
        
        with open(attr_path, 'r') as f:
            self.attr_info = f.readlines()
        
        self.attr_names = self.attr_info[1].split()
        
        self.attr_df = pd.DataFrame([line.split() for line in self.attr_info[2:]], 
                                    columns=['image_id'] + self.attr_names)
        
        for attr in self.attr_names:
            self.attr_df[attr] = self.attr_df[attr].astype(int)
        
        self.df = pd.merge(self.partition_df, self.attr_df, on='image_id')
        
        self.target_idx = self.attr_names.index('Blond_Hair')
        self.bias_idx = self.attr_names.index('Male')
        
        print(f"로드된 {partition_type} 이미지 수: {len(self.df)}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_id']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        target = 1 if int(self.df.iloc[idx][self.attr_names[self.target_idx]]) == 1 else 0
        bias = 1 if int(self.df.iloc[idx][self.attr_names[self.bias_idx]]) == 1 else 0
        
        return image, target, bias, img_name


def extract_celeba_features(data_path, output_path, batch_size=32):
    celeba_dir = os.path.join(data_path, "celebA")
    img_dir = os.path.join(celeba_dir, "Img/img_align_celeba")
    attr_path = os.path.join(celeba_dir, "Anno/list_attr_celeba.txt")
    partition_path = os.path.join(celeba_dir, "Eval/list_eval_partition.txt")

    os.makedirs(output_path, exist_ok=True)


    transform = transforms.Compose([
        transforms.Resize((224, 224)),    # ResNet 표준 크기
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = models.resnet18(pretrained=True)
    backbone.fc = nn.Identity()  # (512-dim features)
    backbone = backbone.to(device)
    backbone.eval()


    for split in ['train', 'val', 'test']:
        print(f"\n🔹 Processing {split} split...")
        dataset = CelebADataset(img_dir, attr_path, partition_path, split, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=2)

        split_out_dir = os.path.join(output_path, split)
        os.makedirs(split_out_dir, exist_ok=True)
        
        print(f"기존 {split} 배치 파일 삭제 중...")
        for f in glob.glob(os.path.join(split_out_dir, '*_batch_*.npy')):
            os.remove(f)
        print("삭제 완료.")
        # ----------------------------------------------------

        with torch.no_grad():
            for i, (batch_images, batch_targets, batch_biases, batch_names) in enumerate(tqdm(dataloader)):
                batch_images = batch_images.to(device)

                batch_features = backbone(batch_images)
                batch_features = batch_features.view(batch_features.size(0), -1) # (B, 512, 1, 1) -> (B, 512)
                batch_features = batch_features.detach().cpu().numpy()

                batch_targets = batch_targets.numpy()
                batch_biases = batch_biases.numpy()

                np.save(os.path.join(split_out_dir, f"feats_batch_{i:05d}.npy"), batch_features)
                np.save(os.path.join(split_out_dir, f"targets_batch_{i:05d}.npy"), batch_targets)
                np.save(os.path.join(split_out_dir, f"bias_batch_{i:05d}.npy"), batch_biases)

                del batch_images, batch_features, batch_targets, batch_biases
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        print(f"✅ {split} split feature 저장 완료 → {split_out_dir}")


def merge_celeba_features(feature_root="./datasets/celeba_features"):
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(feature_root, split)
        print(f"\n🔸 Merging {split} split...")

        feat_list, target_list, bias_list = [], [], []

        feat_files = sorted([f for f in os.listdir(split_dir) if f.startswith('feats_batch_') and f.endswith('.npy')])
        
        if not feat_files:
            print(f"⚠️ 경고: {split} split에 병합할 배치 파일이 없습니다.")
            continue

        for feat_file in tqdm(feat_files):
            idx_str = feat_file.split('_')[-1].split('.')[0] 
            
            feats = np.load(os.path.join(split_dir, feat_file))
            targets = np.load(os.path.join(split_dir, f"targets_batch_{idx_str}.npy"))
            bias = np.load(os.path.join(split_dir, f"bias_batch_{idx_str}.npy"))

            feat_list.append(feats)
            target_list.append(targets)
            bias_list.append(bias)

        feats_all = np.concatenate(feat_list, axis=0)
        targets_all = np.concatenate(target_list, axis=0)
        bias_all = np.concatenate(bias_list, axis=0)

        np.save(os.path.join(feature_root, f"{split}_feats.npy"), feats_all)
        np.save(os.path.join(feature_root, f"{split}_targets.npy"), targets_all)
        np.save(os.path.join(feature_root, f"{split}_bias.npy"), bias_all)

        print(f"✅ {split} 병합 완료 → {feature_root}")
        print(f"   → {split}_feats.npy shape: {feats_all.shape} (정상: {len(feat_list)}개 배치)")
        print(f"   → {split}_targets.npy shape: {targets_all.shape}")
        print(f"   → {split}_bias.npy shape: {bias_all.shape}")



if __name__ == "__main__":
    data_path = "./datasets"
    output_path = "./datasets/celeba_features"
    extract_celeba_features(data_path, output_path)
    merge_celeba_features("./datasets/celeba_features")