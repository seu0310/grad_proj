import sys, types

# --- 🔧 torchvision의 lzma import 우회용 더미 모듈 생성 ---
if 'lzma' not in sys.modules:
    fake_lzma = types.SimpleNamespace()
    fake_lzma.open = lambda *args, **kwargs: None  # 더미 open 함수 추가
    sys.modules['lzma'] = fake_lzma
# -----------------------------------------------------------

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch.nn as nn

# 🔽 config 추가
import utils.config as config

class WaterbirdsDataset(Dataset):
    def __init__(self, data_dir, metadata_path, split, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.metadata = pd.read_csv(metadata_path)

        split_dict = {'train': 0, 'val': 1, 'test': 2}
        self.metadata = self.metadata[self.metadata['split'] == split_dict[split]]

        # 🔽 config에서 target/bias 컬럼명 불러오기
        self.target_attr = config.target_attribute
        self.bias_attr = config.bias_attribute

        print(f"로드된 {split} 이미지 수: {len(self.metadata)}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_filename = self.metadata.iloc[idx]['img_filename']
        img_path = os.path.join(self.data_dir, img_filename)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 🔽 컬럼명 기반으로 target과 bias 읽기
        target = int(self.metadata.iloc[idx][self.target_attr])
        bias = int(self.metadata.iloc[idx][self.bias_attr])

        return image, target, bias

def extract_waterbirds_features(data_path, output_path, batch_size=32):
    # 경로 설정
    waterbirds_dir = os.path.join(data_path, "waterbirds")
    waterbirds_dataset_dir = os.path.join(waterbirds_dir, "waterbird_complete95_forest2water2")
    metadata_path = os.path.join(waterbirds_dataset_dir, "metadata.csv")

    os.makedirs(output_path, exist_ok=True)

    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ✅ pretrained ResNet-18 로드 (마지막 FC 제거)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = models.resnet18(pretrained=True)
    backbone.fc = nn.Identity()  # (512-dim features)
    backbone = backbone.to(device)
    backbone.eval()

    for split in ['train', 'val', 'test']:
        print(f"Processing {split} split...")

        dataset = WaterbirdsDataset(waterbirds_dataset_dir, metadata_path, split, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        features_list = []
        targets_list = []
        biases_list = []

        with torch.no_grad():
            for batch_images, batch_targets, batch_biases in tqdm(dataloader):
                batch_images = batch_images.to(device)

                # 이미지 → feature 추출
                feats = backbone(batch_images)  # shape: (B, 512)
                feats = feats.view(feats.size(0), -1)

                features_list.append(feats.cpu().numpy())
                targets_list.append(batch_targets.numpy())
                biases_list.append(batch_biases.numpy())

        # 저장
        all_features = np.concatenate(features_list, axis=0)
        all_targets = np.concatenate(targets_list, axis=0)
        all_biases = np.concatenate(biases_list, axis=0)

        np.save(os.path.join(output_path, f"{split}_feats.npy"), all_features)
        np.save(os.path.join(output_path, f"{split}_targets.npy"), all_targets)
        np.save(os.path.join(output_path, f"{split}_bias.npy"), all_biases)

        print(f"Saved {split} features of shape {all_features.shape}")

if __name__ == "__main__":
    data_path = "./datasets"  # waterbirds 폴더가 datasets/waterbirds에 있다고 가정
    output_path = "./datasets/waterbirds_features"
    extract_waterbirds_features(data_path, output_path)