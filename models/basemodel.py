import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    # Baseline Network
    def __init__(self, model_name, num_classes, mlp_neurons=None, hid_dim=None):
        super(Network, self).__init__()
        self.num_classes = num_classes
        self.layer_1 = nn.Sequential(
                        nn.Linear(hid_dim, mlp_neurons),
                        nn.Tanh()
                    )
        self.classifier = nn.Linear(mlp_neurons, self.num_classes+1)

    def forward(self, feats_x):
        # feats_x are the previously stored blackbox features
        x1 = self.layer_1(feats_x)
        logits = self.classifier(x1)
        probas = nn.Softmax(dim=1)(logits)
        return logits, probas[:, 1], F.normalize(x1)



class NetworkMargin(nn.Module):
    r"""Large margin arc distance network"""
    def __init__(self, model_name, num_classes, DEVICE, std, mlp_neurons=None, hid_dim=None, easy_margin=None):
        super(NetworkMargin, self).__init__()
        self.num_classes = num_classes
        self.device = DEVICE
        self.std = std
        self.easy_margin = easy_margin

        # 기본값 설정
        if hid_dim is None:
            hid_dim = 512
        if mlp_neurons is None:
            mlp_neurons = 128

        self.new_feats = nn.Sequential(
            nn.Linear(hid_dim, mlp_neurons),
            nn.ReLU(),
        )

        self.weight1 = nn.Parameter(torch.FloatTensor(num_classes + 1, mlp_neurons))
        nn.init.xavier_uniform_(self.weight1)

    def forward(self, feats_x, m=None, s=None):
        # scale 지정
        self.s = s if s is not None else getattr(self, 's', 8)

        # feature 변환
        x = self.new_feats(feats_x)

        # cosine similarity
        cosine = F.linear(F.normalize(x), F.normalize(self.weight1))
        probas = F.softmax(cosine, dim=1)[:, 1]

        # evaluation 모드에서도 항상 5개 반환
        if not self.training:
            return cosine, probas, F.normalize(x), cosine, None

        # margin 처리
        if m is None:
            m = torch.zeros_like(cosine, device=self.device)
        elif isinstance(m, float):
            m = torch.ones_like(cosine, device=self.device) * m

        m = torch.normal(mean=m, std=self.std)
        m = 1 - m

        cos_m = torch.cos(m)
        sin_m = torch.sin(m)
        th = torch.cos(math.pi - m)
        mm = torch.sin(math.pi - m) * m

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * cos_m - sine * sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > th, phi, cosine - mm)

        output = phi * self.s

        return output, probas, x, cosine, None