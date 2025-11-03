import os
import tarfile
import gdown

def download_waterbirds(data_path):
    print("Downloading Waterbirds dataset")
    water_birds_dir = os.path.join(data_path, "waterbirds")
    os.makedirs(water_birds_dir, exist_ok=True)
    
    water_birds_dir_tar = os.path.join(water_birds_dir, "waterbirds.tar.gz")
    gdown.download(
        "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz",
        water_birds_dir_tar,
    )
    
    # 압축 해제
    print("Extracting files...")
    tar = tarfile.open(water_birds_dir_tar, "r:gz")
    tar.extractall(water_birds_dir)
    tar.close()
    
    # 다운로드된 파일 삭제 (선택사항)
    os.remove(water_birds_dir_tar)
    
    print("Waterbirds dataset downloaded successfully!")

if __name__ == "__main__":
    data_path = "./datasets"  # 데이터셋을 저장할 기본 경로
    download_waterbirds(data_path)
