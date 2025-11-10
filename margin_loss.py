import warnings
warnings.filterwarnings("ignore", message="Could not import the lzma module")

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.config as config
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.clustering import get_margins, obtain_and_evaluate_clusters
from utils.dataset import CelebaDataset, WaterBirds
from utils.utils import compute_accuracy, save_state_dict, demographic_parity_difference, equal_opportunity_difference, equalized_odds_difference
import glob

from models.basemodel import Network, NetworkMargin


def parse_args():
    # Parse the arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='baseline',
                        help='baseline or adversarial')
    parser.add_argument('--dataset', type=str, default='celeba',
                        help='which dataset to train on?')   
    parser.add_argument('--bias', action='store_true',
                        help='bias-amplify the model?')           
    parser.add_argument('--clustering', action='store_true',
                        help='only cluster')
    parser.add_argument('--train', action='store_true',
                        help='train, eval, test')
    parser.add_argument('--val-only', action='store_true',
                        help='evaluate on the val set one time')
    parser.add_argument('--test-only', action='store_true',
                        help='evaluate on the test set one time')
    parser.add_argument("--gpu", type=str, default='0',
                        help='gpu card ID')
    parser.add_argument('--seed', type=int, default=4594,
                        help='seed to run')
    args = parser.parse_args()
    return args


def read_data(args):
    # Create the train, test and val loaders
    
    batch_size = config.base_batch_size
    if args.train:
        if args.dataset == 'celeba':
            train_dataset = CelebaDataset(split=0)
            valid_dataset = CelebaDataset(split=1)
            test_dataset = CelebaDataset(split=2)

        elif args.dataset == 'waterbirds':
            train_dataset = WaterBirds(split='train')
            valid_dataset = WaterBirds(split='val')
            test_dataset = WaterBirds(split='test')

        if args.bias:
            class_sample_count = train_dataset.class_sample_count
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in train_dataset.targets])
            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            print(samples_weight.shape)
            shuffle = False
            sampler = sampler
        else:
            shuffle = True
            sampler = None
            
        train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            sampler=sampler,
                            num_workers=0)

        valid_loader = DataLoader(dataset=valid_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)

        test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)
        
        return train_loader, valid_loader, test_loader

    elif args.val_only:
        if args.dataset == 'celeba':
            valid_dataset = CelebaDataset(split=1)
        else:
            valid_dataset = WaterBirds(split='val')
        valid_loader = DataLoader(dataset=valid_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)
        return valid_loader
    
    else:
        if args.dataset == 'celeba':
            test_dataset = CelebaDataset(split=2)
        else:
            test_dataset = WaterBirds(split='test')
        test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)
        return test_loader

class GeneralizedCELoss(nn.Module):

    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
             
    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach()**self.q)*self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')
  
        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight
        return loss


def cross_entropy_loss_arc(logits, labels, **kwargs):
    """ Modified cross entropy loss to compute the margin loss"""
    nll = F.log_softmax(logits, dim=-1)
    loss = -nll * labels
    return loss.sum(dim=-1).mean()


def train(model, NUM_EPOCHS, optimizer, DEVICE, train_loader, valid_loader, test_loader, args):
    # Margin용 baseline, kmeans 준비
    if args.type == 'margin':
        baseline = Network(config.model_name, config.num_class, config.mlp_neurons, config.hid_dim)
        model_name = config.basemodel_path
        with torch.no_grad():
            baseline.load_state_dict(torch.load(os.path.join('./', model_name), map_location=DEVICE))
        baseline.eval()
        baseline = baseline.to(DEVICE)
        kmeans, _, all_margins = get_margins(train_loader, baseline, DEVICE)

    start_time = time.time()
    best_val = 0
    best_worst, best_avg = 999, 999
    final_epoch = 0
    val_acc_list = []    # Top-K ensemble by val accuracy
    val_worst_list = []  # Top-K ensemble by worst-group accuracy

    os.makedirs('ensemble_model', exist_ok=True)

    for epoch in range(NUM_EPOCHS):

        model.train()
        for _, (_, features, targets, z1, _) in enumerate(train_loader):
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            z1 = z1.to(DEVICE)

            if args.type == 'margin':
                one_hot = F.one_hot(targets).to(DEVICE)
                with torch.no_grad():
                    _, _, feats_baseline = baseline(features)
                feats_baseline = feats_baseline.cpu().detach().numpy()
                pseudo_labels = kmeans.predict(feats_baseline)

                margins = all_margins[pseudo_labels]
                margins = torch.from_numpy(margins).to(DEVICE)
                features = features.to(torch.float32)
                logits, _, _, _, _ = model(features, margins, s=config.scale)
                cost = cross_entropy_loss_arc(logits, one_hot.float())

            elif args.type == 'baseline':
                logits, _, _ = model(features)
                cost = nn.CrossEntropyLoss()(logits, targets.long())

            optimizer.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()


        model.eval()
        with torch.set_grad_enabled(False):
            if args.type == 'margin':
                train_acc, train_worst, train_avg = compute_accuracy(model, train_loader, device=DEVICE, margin=True)
                val_acc, val_worst, val_avg = compute_accuracy(model, valid_loader, device=DEVICE, margin=True)
                test_acc, test_worst, test_avg = compute_accuracy(model, test_loader, device=DEVICE, margin=True)
            else:
                train_acc, train_worst, train_avg = compute_accuracy(model, train_loader, device=DEVICE)
                val_acc, val_worst, val_avg = compute_accuracy(model, valid_loader, device=DEVICE)
                test_acc, test_worst, test_avg = compute_accuracy(model, test_loader, device=DEVICE)

            overall_acc = train_acc if (args.type=='baseline' and args.bias) else val_acc



            # Create seed-specific directories
            best_model_dir = f'saved_models/seed_{args.seed}'
            ensemble_model_dir = f'ensemble_model/seed_{args.seed}'
            os.makedirs(best_model_dir, exist_ok=True)
            os.makedirs(ensemble_model_dir, exist_ok=True)

            if best_val < overall_acc:
                #print(f'Best model saved at epoch {epoch}')
                final_epoch = epoch
                best_val = overall_acc
                best_worst = test_worst
                best_avg = test_avg
                
                original_path = config.margin_path if args.type=='margin' else config.basemodel_path
                path = os.path.join(best_model_dir, original_path)
                save_state_dict(model.state_dict(), path)

            epoch_path = os.path.join(ensemble_model_dir, f'{args.type}_epoch{epoch}.pt')
            save_state_dict(model.state_dict(), epoch_path)

            val_acc_list.append(val_acc if isinstance(val_acc, float) else val_acc.cpu().item())
            val_worst_list.append(float(val_worst))
            save_path_acc = os.path.join(ensemble_model_dir, f'{args.type}_val_accs.npy')
            save_path_worst = os.path.join(ensemble_model_dir, f'{args.type}_val_worsts.npy')
            np.save(save_path_acc, np.array(val_acc_list))
            np.save(save_path_worst, np.array(val_worst_list))


            print(f'Epoch {epoch}')
            print('Train worst, avg, global acc:', train_worst, train_avg, train_acc)
            print('Val worst, avg, global acc:', val_worst, val_avg, val_acc)
            print('Test worst, avg, global acc:', test_worst, test_avg, test_acc)

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    print("Final val acc:", best_val)
    print("Test Worst:", best_worst)
    print("Test Avg:", best_avg)
    print("Final epoch:", final_epoch)


    ### === 공정성 지표 계산 (margin 모델 기준) === ###
    print("\n[공정성 지표 계산 - Margin 모델]")
    all_preds, all_targets, all_sensitive = [], [], []
    model.eval()
    with torch.no_grad():
        for _, (_, features, targets, z1, _) in enumerate(test_loader):
            features = features.to(DEVICE)
            if args.type == 'margin':
                logits, _, _, _, _ = model(features, m=None, s=None)
            else:
                logits, _, _ = model(features)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(targets)
            all_sensitive.append(z1)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_sensitive = torch.cat(all_sensitive)
    dp = demographic_parity_difference(all_preds, all_sensitive)
    eo = equal_opportunity_difference(all_targets, all_preds, all_sensitive)
    eod = equalized_odds_difference(all_targets, all_preds, all_sensitive)
    print(f"Demographic Parity Diff: {dp:.4f}")
    print(f"Equal Opportunity Diff: {eo:.4f}")
    print(f"Equalized Odds Diff: {eod:.4f}")
    ### ===================================== ###


    return best_val



def eval(model, data_loader, path):
    
    model.load_state_dict(torch.load(os.path.join('./', path), map_location=DEVICE)) 
    model.eval()

    with torch.no_grad():
        test_acc, test_worst, test_avg = compute_accuracy(model, data_loader, DEVICE)
        print("Global Acc", test_acc)
        print("Worst:", test_worst)
        print("Avg:", test_avg)



def eval_ensemble_topk(model_class, data_loader, ckpt_paths,
                       val_accs, val_worsts, device,
                       topk=5, metric='worst', model_type='margin'):

    metric_arr = val_accs if metric == 'val' else val_worsts
    topk_indices = np.argsort(metric_arr)[-topk:]
    ckpt_paths_topk = [ckpt_paths[i] for i in topk_indices]

    print(f"Evaluating ensemble with Top-{topk} models "
          f"(metric={metric}, from {len(ckpt_paths)} checkpoints)...")

    models = []
    for path in ckpt_paths_topk:
        m = model_class()
        m.load_state_dict(torch.load(path, map_location=device))
        m.to(device)
        m.eval()
        models.append(m)

    all_logits_list, all_targets_list, all_sensitive_list = [], [], []

    with torch.no_grad():
        for _, (_, features, targets, z1, _) in enumerate(data_loader):
            features = features.to(device)
            logits_list = []
            for m in models:
                if model_type == 'baseline':
                    logits, _, _ = m(features)
                else:
                    logits, _, _, _, _ = m(features, m=None, s=None)
                logits_list.append(logits)

            # 모델별 logits 평균
            avg_logits = torch.stack(logits_list).mean(dim=0)
            all_logits_list.append(avg_logits)
            all_targets_list.append(targets)
            all_sensitive_list.append(z1)

    # === Concatenate 전체 데이터 ===
    all_logits = torch.cat([l.cpu() for l in all_logits_list], dim=0)
    all_targets = torch.cat(all_targets_list, dim=0)
    all_sensitive = torch.cat(all_sensitive_list, dim=0)
    preds = all_logits.argmax(dim=1)

    # 전체 / 그룹별 정확도 계산
    global_acc = (preds == all_targets).float().mean().item()
    group_ids = torch.tensor(data_loader.dataset.group_ids)
    group_accs = [(preds[group_ids == g] == all_targets[group_ids == g]).float().mean().item()
                  for g in torch.unique(group_ids)]
    worst_acc = min(group_accs)
    avg_acc = sum(group_accs) / len(group_accs)

    ### === 공정성 지표 계산 (Ensemble 모델 기준) === ###
    print("\n[공정성 지표 계산 - Ensemble 모델]")
    dp = demographic_parity_difference(preds, all_sensitive)
    eo = equal_opportunity_difference(all_targets, preds, all_sensitive)
    eod = equalized_odds_difference(all_targets, preds, all_sensitive)
    print(f"Demographic Parity Diff: {dp:.4f}")
    print(f"Equal Opportunity Diff: {eo:.4f}")
    print(f"Equalized Odds Diff: {eod:.4f}")
    ### ===================================== ###

    print(f"Top-{topk} Ensemble Global Acc:", global_acc)
    print(f"Top-{topk} Ensemble Worst Acc:", worst_acc)
    print(f"Top-{topk} Ensemble Avg Acc:", avg_acc)
    return global_acc, worst_acc, avg_acc



def main(args):
    seed = args.seed
    
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    DEVICE = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    if args.dataset == 'celeba' or args.dataset == 'waterbirds':
        celeba = True
    else:
        celeba = False
    
    if args.train:
        # For training
        train_loader, valid_loader, test_loader = read_data(args)
        if args.type == 'baseline':
            # Baseline training
            model = Network(config.model_name, config.num_class, config.mlp_neurons, config.hid_dim)
            model.to(DEVICE)
            lr = config.base_lr
            weight_decay = config.weight_decay
            if config.opt_b == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            epochs = config.base_epochs
            train(model, config.base_epochs, optimizer, DEVICE, train_loader, valid_loader, test_loader, args)
        elif args.type == 'margin':
            # Margin loss
            model = NetworkMargin(config.model_name, config.num_class, DEVICE, config.std, config.mlp_neurons, config.hid_dim)
            model = model.to(DEVICE)
            lr = config.base_lr
            weight_decay = config.weight_decay
            if config.opt_m == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            train(model, config.base_epochs, optimizer, DEVICE, train_loader, valid_loader, test_loader, args)
        
    elif args.clustering:
        # Calculate cluster NMIs
        args.train = True
        train_loader, valid_loader, test_loader = read_data(args)

        baseline = Network(config.model_name, config.num_class, config.mlp_neurons, config.hid_dim)
        model_name = config.basemodel_path
        with torch.no_grad():
            
            ''' Comment only if you do not want to load '''
            baseline.load_state_dict(torch.load(os.path.join('./', model_name), map_location=DEVICE))
            baseline.to(DEVICE)
            baseline.eval()
            obtain_and_evaluate_clusters(train_loader, baseline, DEVICE)

    elif args.val_only:
        valid_loader = read_data(args)
        
        if args.type == 'baseline':
            model = Network(config.model_name, config.num_class, config.mlp_neurons)
        else:
            model = NetworkMargin(config.model_name, config.num_class, DEVICE, config.mlp_neurons)
        
        model = model.to(DEVICE)
        
        if args.type == 'baseline':
            eval(model, valid_loader, config.basemodel_path)
        else:
            eval(model, valid_loader, config.margin_path)

    elif args.test_only:
        test_loader = read_data(args)
        
        ensemble_model_dir = f'ensemble_model/seed_{args.seed}'

        if args.type == 'baseline':
            ckpt_paths = [os.path.join(ensemble_model_dir, f'baseline_epoch{i}.pt') for i in range(config.base_epochs)]
            model_class = lambda: Network(
                config.model_name, 
                config.num_class, 
                config.mlp_neurons, 
                config.hid_dim
            )
        else:  # margin
            ckpt_paths = [os.path.join(ensemble_model_dir, f'margin_epoch{i}.pt') for i in range(config.base_epochs)]
            model_class = lambda: NetworkMargin(
                config.model_name, 
                config.num_class, 
                DEVICE, 
                std=config.std, 
                mlp_neurons=config.mlp_neurons, 
                hid_dim=config.hid_dim
            )

        val_accs_path = os.path.join(ensemble_model_dir, f'{args.type}_val_accs.npy')
        val_worst_path = os.path.join(ensemble_model_dir, f'{args.type}_val_worsts.npy')
        if not os.path.exists(val_accs_path) or not os.path.exists(val_worst_path):
            raise FileNotFoundError(f"Validation metric files not found in {ensemble_model_dir}. Need both val_accs.npy and val_worsts.npy from training.")
        val_accs = np.load(val_accs_path)
        val_worsts = np.load(val_worst_path)


        global_acc, worst_acc, avg_acc = eval_ensemble_topk(
            model_class,
            test_loader,
            ckpt_paths,
            val_accs,
            val_worsts,
            DEVICE,
            topk=5,
            metric="worst",
            model_type=args.type
        )
    
    print("VRAM taken: ", torch.cuda.max_memory_allocated() / 1024**2)


if __name__ == '__main__':
    args = parse_args()
    main(args)