import os
import gdown
from zipfile import ZipFile

def download_celeba(data_path):
    print("Downloading CelebA dataset")
    celeba_dir = os.path.join(data_path, "celeba")
    os.makedirs(celeba_dir, exist_ok=True)
    
    # 이미지 다운로드
    gdown.download(
        "https://drive.google.com/uc?id=1mb1R6dXfWbvk3DnlWOBO8pDeoBKOcLE6",
        os.path.join(celeba_dir, "img_align_celeba.zip"),
    )
    
    # 속성 데이터 다운로드
    gdown.download(
        "https://drive.google.com/uc?id=1acn0-nE4W7Wa17sIkKB0GtfW4Z41CMFB",
        os.path.join(celeba_dir, "list_eval_partition.txt"),
        quiet=False
    )
    
    gdown.download(
        "https://drive.google.com/uc?id=11um21kRUuaUNoMl59TCe2fb01FNjqNms",
        os.path.join(celeba_dir, "list_attr_celeba.txt"),
        quiet=False
    )
    
    # 이미지 압축 해제
    print("Extracting images...")
    with ZipFile(os.path.join(celeba_dir, "img_align_celeba.zip"), 'r') as zip_ref:
        zip_ref.extractall(celeba_dir)
    
    print("CelebA dataset downloaded successfully!")
    
if __name__ == "__main__":
    data_path = "./datasets"  # 데이터셋을 저장할 기본 경로
    download_celeba(data_path)
