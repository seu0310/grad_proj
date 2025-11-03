# Mitigating Biases in Blackbox Feature Extractors for Image Classification Tasks

본 리포지토리는 졸업 프로젝트를 위한 수정 된 파일들의 리포지토리입니다
	
</br>

## Dataset

* Download the CelebA, and Waterbirds datasets. Extract their features from a pretrained feature extractor and store them. Note the folder path and update it in `utils/config.py`. Recall that the validation set has to be group-balanced. For CelebA, store the folder in the `celeba_path` attribute of `utils/config.py` (similarly for Waterbirds). The embeddings should be stored in `{split}_feats.npy`, whereas the bias and target labels should be stored in `{split}_bias.npy` and `{split}_target.npy` respectively, inside the folder. 

</br>

## 

### Baseline
To run the baseline (ERM model), run the following command. Dataset can either be waterbirds or celeba:

```bash
python margin_loss.py --dataset waterbirds --train --type baseline --bias 
```

### Margin loss (CAML) training
```bash
python margin_loss.py --dataset waterbirds --train --type margin
```

</br>
