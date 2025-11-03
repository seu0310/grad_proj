import os
import subprocess

def main():
    # 필요한 디렉토리 생성
    #os.makedirs("./datasets", exist_ok=True)
    #os.makedirs("./utils", exist_ok=True)
    
    #print("1. 데이터셋 다운로드 중...")
    #subprocess.run(["python", "setup_celeba.py"])
    #subprocess.run(["python", "setup_waterbirds.py"])
    
    #print("2. 특성 추출 중...")
    subprocess.run(["python", "extract_celeba_features.py"])
    #subprocess.run(["python", "extract_waterbirds_features.py"])
    
    #print("3. 검증 세트 균형 맞추기...")
    subprocess.run(["python", "balance_validation_sets.py"])

    print("모든 준비 완료!")

if __name__ == "__main__":
    main()
