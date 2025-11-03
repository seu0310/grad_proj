import os
import numpy as np
from sklearn.model_selection import train_test_split

def balance_celeba_validation(data_path):
    print("Balancing CelebA validation set...")
    
    # 기존 특성 및 레이블 로드
    val_feats = np.load(os.path.join(data_path, "val_feats.npy"))
    val_targets = np.load(os.path.join(data_path, "val_targets.npy"))
    val_bias = np.load(os.path.join(data_path, "val_bias.npy"))
    
    # 그룹 정의: (target, bias) 조합
    groups = []
    for i in range(len(val_targets)):
        groups.append((val_targets[i], val_bias[i]))
    groups = np.array(groups)
    
    # 고유 그룹 찾기
    unique_groups = np.unique(groups, axis=0)
    
    # 각 그룹의 인덱스 찾기
    group_indices = {}
    for group in unique_groups:
        group_tuple = tuple(group)
        group_indices[group_tuple] = np.where((groups == group).all(axis=1))[0]
    
    # 그룹별 샘플 수 세기
    for group, indices in group_indices.items():
        print(f"그룹 {group}: {len(indices)} 샘플")
    
    # 최소 그룹 크기 찾기
    min_group_size = min(len(indices) for indices in group_indices.values())
    
    # 각 그룹에서 동일한 수의 샘플 선택
    balanced_indices = []
    for indices in group_indices.values():
        if len(indices) > min_group_size:
            selected_indices = np.random.choice(indices, min_group_size, replace=False)
        else:
            selected_indices = indices
        balanced_indices.extend(selected_indices)
    
    # 새로운 균형 잡힌 검증 세트 생성
    balanced_val_feats = val_feats[balanced_indices]
    balanced_val_targets = val_targets[balanced_indices]
    balanced_val_bias = val_bias[balanced_indices]
    
    # 기존 파일 백업
    os.rename(os.path.join(data_path, "val_feats.npy"), os.path.join(data_path, "val_feats_original.npy"))
    os.rename(os.path.join(data_path, "val_targets.npy"), os.path.join(data_path, "val_targets_original.npy"))
    os.rename(os.path.join(data_path, "val_bias.npy"), os.path.join(data_path, "val_bias_original.npy"))
    
    # 새로운 균형 잡힌 데이터 저장
    np.save(os.path.join(data_path, "val_feats.npy"), balanced_val_feats)
    np.save(os.path.join(data_path, "val_targets.npy"), balanced_val_targets)
    np.save(os.path.join(data_path, "val_bias.npy"), balanced_val_bias)
    
    print(f"균형 잡힌 검증 세트 크기: {len(balanced_val_feats)}")

def balance_waterbirds_validation(data_path):
    print("Balancing Waterbirds validation set...")
    
    # 위의 CelebA 코드와 동일한 로직 적용
    val_feats = np.load(os.path.join(data_path, "val_feats.npy"))
    val_targets = np.load(os.path.join(data_path, "val_targets.npy"))
    val_bias = np.load(os.path.join(data_path, "val_bias.npy"))
    
    # 그룹 정의: (targets, bias) 조합
    groups = []
    for i in range(len(val_targets)):
        groups.append((val_targets[i], val_bias[i]))
    groups = np.array(groups)
    
    # 고유 그룹 찾기
    unique_groups = np.unique(groups, axis=0)
    
    # 각 그룹의 인덱스 찾기
    group_indices = {}
    for group in unique_groups:
        group_tuple = tuple(group)
        group_indices[group_tuple] = np.where((groups == group).all(axis=1))[0]
    
    # 그룹별 샘플 수 세기
    for group, indices in group_indices.items():
        print(f"그룹 {group}: {len(indices)} 샘플")
    
    # 최소 그룹 크기 찾기
    min_group_size = min(len(indices) for indices in group_indices.values())
    
    # 각 그룹에서 동일한 수의 샘플 선택
    balanced_indices = []
    for indices in group_indices.values():
        if len(indices) > min_group_size:
            selected_indices = np.random.choice(indices, min_group_size, replace=False)
        else:
            selected_indices = indices
        balanced_indices.extend(selected_indices)
    
    # 새로운 균형 잡힌 검증 세트 생성
    balanced_val_feats = val_feats[balanced_indices]
    balanced_val_targets = val_targets[balanced_indices]
    balanced_val_bias = val_bias[balanced_indices]
    
    # 기존 파일 백업
    os.rename(os.path.join(data_path, "val_feats.npy"), os.path.join(data_path, "val_feats_original.npy"))
    os.rename(os.path.join(data_path, "val_targets.npy"), os.path.join(data_path, "val_targets_original.npy"))
    os.rename(os.path.join(data_path, "val_bias.npy"), os.path.join(data_path, "val_bias_original.npy"))
    
    # 새로운 균형 잡힌 데이터 저장
    np.save(os.path.join(data_path, "val_feats.npy"), balanced_val_feats)
    np.save(os.path.join(data_path, "val_targets.npy"), balanced_val_targets)
    np.save(os.path.join(data_path, "val_bias.npy"), balanced_val_bias)
    
    print(f"균형 잡힌 검증 세트 크기: {len(balanced_val_feats)}")

if __name__ == "__main__":
    balance_celeba_validation("./datasets/celeba_features")
    #balance_waterbirds_validation("./datasets/waterbirds_features")
