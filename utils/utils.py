import numpy as np
import torch





def _to_numpy(x):
    """torch.Tensor → numpy 변환 (자동 감지)"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def demographic_parity_difference(y_pred, sensitive_attr):
    """
    📊 Demographic Parity Difference (DPD)
    그룹별 예측 긍정 비율(P(Ŷ=1|A))의 차이
    
    y_pred: torch.Tensor or np.array (0/1)
    sensitive_attr: torch.Tensor or np.array (예: 성별, 인종 등)
    """
    y_pred = _to_numpy(y_pred)
    sensitive_attr = _to_numpy(sensitive_attr)

    groups = np.unique(sensitive_attr)
    if len(groups) != 2:
        raise ValueError("Demographic Parity는 현재 이진 그룹만 지원합니다.")

    p_y1_g0 = np.mean(y_pred[sensitive_attr == groups[0]])
    p_y1_g1 = np.mean(y_pred[sensitive_attr == groups[1]])
    dp_diff = abs(p_y1_g0 - p_y1_g1)
    return dp_diff


def equal_opportunity_difference(y_true, y_pred, sensitive_attr):
    """
    🎯 Equal Opportunity Difference (EOD)
    실제 긍정(Y=1) 중에서 예측도 긍정인 비율(TPR)의 차이
    
    y_true: torch.Tensor or np.array (0/1)
    y_pred: torch.Tensor or np.array (0/1)
    sensitive_attr: torch.Tensor or np.array (예: 성별, 인종 등)
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    sensitive_attr = _to_numpy(sensitive_attr)

    groups = np.unique(sensitive_attr)
    if len(groups) != 2:
        raise ValueError("Equal Opportunity는 현재 이진 그룹만 지원합니다.")

    # True Positive Rate (TPR)
    tpr_g0 = np.mean(y_pred[(sensitive_attr == groups[0]) & (y_true == 1)])
    tpr_g1 = np.mean(y_pred[(sensitive_attr == groups[1]) & (y_true == 1)])
    eo_diff = abs(tpr_g0 - tpr_g1)
    return eo_diff


def equalized_odds_difference(y_true, y_pred, sensitive_attr):
    """
    ⚖️ Equalized Odds Difference (EODs)
    TPR (True Positive Rate)와 FPR (False Positive Rate) 둘 다 비슷해야 함.
    두 지표의 평균 차이를 반환.
    
    y_true: torch.Tensor or np.array (0/1)
    y_pred: torch.Tensor or np.array (0/1)
    sensitive_attr: torch.Tensor or np.array (예: 성별, 인종 등)
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    sensitive_attr = _to_numpy(sensitive_attr)

    groups = np.unique(sensitive_attr)
    if len(groups) != 2:
        raise ValueError("Equalized Odds는 현재 이진 그룹만 지원합니다.")

    # Group 0
    tpr_g0 = np.mean(y_pred[(sensitive_attr == groups[0]) & (y_true == 1)])
    fpr_g0 = np.mean(y_pred[(sensitive_attr == groups[0]) & (y_true == 0)])
    # Group 1
    tpr_g1 = np.mean(y_pred[(sensitive_attr == groups[1]) & (y_true == 1)])
    fpr_g1 = np.mean(y_pred[(sensitive_attr == groups[1]) & (y_true == 0)])

    tpr_diff = abs(tpr_g0 - tpr_g1)
    fpr_diff = abs(fpr_g0 - fpr_g1)
    eod_diff = (tpr_diff + fpr_diff) / 2.0  # 평균 차이
    return eod_diff





def biased_acc(y, y_, u):
    # Computes worst and avg accuracies
    g = np.zeros([2, 2])
    uc = np.zeros([2, 2])
    for i in range(u.shape[0]):
        if u[i] > 0:
            g[int(y[i])][1] += y_[i]
            uc[int(y[i])][1] += 1
        else:
            g[int(y[i])][0] += y_[i]
            uc[int(y[i])][0] += 1
    acc = g / uc
    acc[0, :] = 1 - acc[0, :]
    worst = np.min(acc)
    avg = np.mean(acc)
    #print(acc[0, 0], acc[0, 1], acc[1, 0], acc[1, 1])
    return worst, avg


def save_state_dict(state_dict, save_path):
    # Saves model
    torch.save(state_dict, save_path)


def compute_accuracy(model, data_loader, device, margin=False):

    correct_pred, num_examples = 0, 0
    pred_total = []
    y_total = []
    gen_total = []

    for _, (_, features, targets, gender, _) in enumerate(data_loader):
        features = features.to(device).to(torch.float32)
        targets = targets.to(device)
        gender = gender.to(device)
        y = targets.cpu().detach().numpy()


        if margin:
            logits, _, _, _, _ = model(features, m=None, s=None)
        else:
            logits, _, _ = model(features)


        probas = torch.softmax(logits, dim=1)[:,1]  # margin/baseline 모두 2차원 [batch,2]로 가정
        predicted_labels = (probas >= 0.5).int().squeeze()

        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()

        class_pred = predicted_labels.cpu().detach().numpy()
        gen = gender.cpu().detach().numpy()

        pred_total += class_pred.tolist()
        gen_total += gen.tolist()
        y_total += y.tolist()


    worst, avg = biased_acc(np.array(y_total), np.array(pred_total), np.array(gen_total))

    return correct_pred.float()/num_examples * 100, worst, avg