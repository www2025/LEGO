import sys
import argparse
import torch
from scipy.stats import entropy
import torch.nn.functional as F
import torch.nn as nn
# pyg imports
from torch_geometric.datasets import Planetoid, WikiCS, Amazon, Coauthor, LastFMAsia, NELL, CitationFull, GNNBenchmarkDataset
from torch_geometric.transforms import NormalizeFeatures, Compose
from torch_geometric.utils import homophily
sys.path.append("../..")
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import get_acc, get_ood_performance, get_f1_score
from OOD.utils import EntropyLoss, get_consistent_loss_new, cosine_similarity, CE_uniform, seed_torch
from data_process import generate_masks_lego
from sampling_methods import *
from model import OODGAT
import numpy as np

from basic_gnns import GCNDetector
from torch_geometric.utils import to_dense_adj
import warnings
warnings.filterwarnings("ignore")
import math

def train(args):

    dataset_str = args.dataset.split('_')[0]
    # transforms
    trans = []
    continuous = args.continuous
    if not continuous:
        trans.append(NormalizeFeatures())

    trans = Compose(trans)

    if dataset_str == 'LastFMAsia':
        dataset = LastFMAsia(root='pyg_data/LastFMAsia', transform=trans)
    elif dataset_str == 'Amazon':
        dataset_name = args.dataset.split('_')[1]
        dataset = Amazon(root='pyg_data/Amazon', name=dataset_name, transform=trans)
    elif dataset_str == 'GNNBenchmark':
        dataset_name = args.dataset.split('_')[1]
        dataset = GNNBenchmarkDataset(root='pyg_data/GNNBenchmark', name=dataset_name, transform=trans)
    elif dataset_str == 'Planetoid':
        dataset_name = args.dataset.split('_')[1]
        dataset = Planetoid(root='pyg_data/Planetoid', name=dataset_name, transform=trans)
    else:
        raise Exception('unknown dataset.')

    data = dataset[0]
    print()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    print(f'homophily: {homophily(data.edge_index, data.y)}')

    # dataset parameters
    ID_classes = args.ID_classes
    splits = args.splits
    n_samples_init = splits[0]
    n_samples_per_class = splits[1]
    val_size_per_class = splits[2]
    test_size = splits[3]
    
    # #calculate the number of nodes in each class
    # node_labels = data.y
    # class_counts = {cls: 0 for cls in ID_classes}
    # for cls in ID_classes:
    #     class_counts[cls] = (node_labels == cls).sum().item()
    # for cls, count in class_counts.items():
    #     print(f'Class {cls}: {count} nodes')
    
    ID_mask, OOD_mask, train_mask_ID, val_mask_ID, test_mask_ID, train_mask_OOD, val_mask_OOD, test_mask_OOD, \
    detection_mask_val, detection_y_val, joint_y_val, detection_mask_test, detection_y_test, joint_y_test, ID_y, left_idx_all, detection_mask_train, joint_y_train, joint_y_all = \
        generate_masks_lego(data, n_samples_init, ID_classes, n_samples_per_class, val_size_per_class, test_size, args.random_seed_data)

    assert (ID_mask | OOD_mask).sum() == data.num_nodes
    assert (train_mask_ID | val_mask_ID | test_mask_ID | train_mask_OOD | val_mask_OOD | test_mask_OOD).sum() == \
           train_mask_ID.sum() + val_mask_ID.sum() + test_mask_ID.sum() + train_mask_OOD.sum() + val_mask_OOD.sum() + test_mask_OOD.sum()
    assert detection_mask_val.sum() == len(detection_y_val) == len(joint_y_val)
    assert detection_mask_test.sum() == len(detection_y_test) == len(joint_y_test)
    # assert train_mask_ID.sum() == train_mask_OOD.sum() == n_samples_per_class * len(ID_classes)
    assert val_mask_ID.sum() == val_mask_OOD.sum() == val_size_per_class * len(ID_classes)
    assert test_mask_ID.sum() == test_mask_OOD.sum() == test_size

    data.train_mask, data.val_mask, data.test_mask = train_mask_ID, val_mask_ID, test_mask_ID
    data.y = ID_y
    data.joint_y_train, data.detection_mask_train = torch.tensor(joint_y_train), detection_mask_train
    data.joint_y_test, data.detection_mask_test = torch.tensor(joint_y_test), detection_mask_test
    data.joint_y_val, data.detection_mask_val = torch.tensor(joint_y_val), detection_mask_val
    data.joint_y_all = torch.tensor(joint_y_all)
    
    print('ID size: {}, OOD size: {}, total size: {}.'.format(ID_mask.sum(), OOD_mask.sum(), data.num_nodes))
    print('train%ID: {:.2%}, val%ID: {:.2%}, test%ID: {:.2%}.'.format(train_mask_ID.sum() / ID_mask.sum(),
                                                                      val_mask_ID.sum() / ID_mask.sum(),
                                                                      test_mask_ID.sum() / ID_mask.sum()))
    print('train%OOD: {:.2%}, val%OOD: {:.2%}, test%OOD: {:.2%}.'.format(train_mask_OOD.sum() / OOD_mask.sum(),
                                                                         val_mask_OOD.sum() / OOD_mask.sum(),
                                                                         test_mask_OOD.sum() / OOD_mask.sum()))
    device = torch.device('cuda')
    data = data.to(device)


    # inline help functions
    def init_oodgat():
        # fitlog.set_rng_seed(args.random_seed_model)
        # seed_torch(args.random_seed_model)
        # init model
        in_dim = data.x.shape[1]
        out_dim = len(ID_classes)
        if args.model_name == 'OODGAT':
            model = OODGAT(in_dim, args.hidden_dim, out_dim, args.heads, False, args.drop_edge, True,
                           args.drop_prob, True, args.drop_input).to(device)
            model2 = GCNDetector(in_dim, args.hidden_dim, out_dim, args.drop_prob, bias=True).to(device)
        else:
            print('unknown model: {}.'.format(args.model_name))
            raise Exception('unknown model')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr)

        return model, model2, optimizer, optimizer2
    

    def train_oodgat(model, optimizer):
        a = torch.tensor(0.9).to(device)
        b = torch.tensor(0.01).to(device)
        # train
        best_t = args.epochs - 1
        best_metric = 0
        patience = 0
        xent = nn.CrossEntropyLoss()
        ent_loss_func = EntropyLoss(reduction=False)
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            loss = torch.zeros(1).to(device)
            embeds, logits, att = model(data, return_attention_weights=True)
            if args.w_consistent is not None and args.w_consistent > 0:
                ent_loss = ent_loss_func(logits)  # ent_loss: N-dim tensor
                cos_loss_1 = get_consistent_loss_new(att[0].T, (ent_loss - ent_loss.mean()) / ent_loss.std(),
                                                        f1=F.sigmoid, f2=F.sigmoid)
                cos_loss_2 = get_consistent_loss_new(att[1].T, (ent_loss - ent_loss.mean()) / ent_loss.std(),
                                                        f1=F.sigmoid, f2=F.sigmoid)
                consistent_loss = 0.5 * (cos_loss_1 + cos_loss_2)
                loss += torch.pow(a, b * epoch) * args.w_consistent * consistent_loss
            if args.w_discrepancy is not None and args.w_discrepancy > 0:
                loss -= torch.pow(a, b * epoch) * args.w_discrepancy * cosine_similarity(att[0].mean(axis=1), att[1].mean(axis=1))
            if args.w_ent is not None and args.w_ent > 0:
                loss += torch.pow(a, b * epoch) * args.w_ent * local_ent_loss(logits, att, len(ID_classes), args.margin)

            sup_loss = xent(logits[data.train_mask], data.y[data.train_mask])
            loss += sup_loss
            loss.backward()
            optimizer.step()
            # validate
            if epoch % 10 == 0:
                model.eval()
                embeds, logits, att = model(data, return_attention_weights=True)
                preds = logits.argmax(axis=1).detach()
                val_acc = get_acc(data.y, preds, data.val_mask)

                ATT = F.sigmoid(torch.hstack([att[0].detach(), att[1].detach()]).mean(axis=1)).cpu()
                auroc, aupr_0, aupr_1, fprs = get_ood_performance(detection_y_val, ATT,
                                                                          detection_mask_val)

                # print('epoch: {}, loss: {}, val_acc: {}, auroc:{}.'.format(epoch + 1, loss.item(), val_acc.item(),
                #                                                                auroc))

                current_metric = val_acc + auroc
                if current_metric > best_metric:
                    best_t = epoch
                    patience = 0
                    best_metric = current_metric
                    torch.save(model.state_dict(), 'best_oodgat.pkl')
                else:
                    patience += 1
                    if patience > 20:
                        break

        return best_metric, best_t

    def evaluate_oodgat(model):
        # evaluate
        model.load_state_dict(torch.load('best_oodgat.pkl'))
        model.eval()
        # classification
        embeds, logits, a = model(data, return_attention_weights=True)
        preds = logits.argmax(axis=1).detach()
        a = [a[0].detach(), a[1].detach()]
        test_acc = get_acc(data.y, preds, data.test_mask)
        print('test_acc:{}'.format(test_acc.item()))

        # OOD detection
        pred_dist = F.softmax(logits, dim=1).detach().cpu()
        ENT = entropy(pred_dist, axis=1)
        ATT = F.sigmoid(torch.hstack([a[0], a[1]]).mean(axis=1)).cpu()

        auroc_ENT, aupr_0_ENT, aupr_1_ENT, fprs_ENT = get_ood_performance(detection_y_test, ENT,
                                                                          detection_mask_test)
        auroc_ATT, aupr_0_ATT, aupr_1_ATT, fprs_ATT = get_ood_performance(detection_y_test, ATT,
                                                                          detection_mask_test)

        print('detection via ENT: auroc:{}, aupr_0:{}, aupr_1:{}, fpr95:{}'.format(auroc_ENT, aupr_0_ENT, aupr_1_ENT,
                                                                                 fprs_ENT[2]))
        # print('detection via ATT: auroc:{}, aupr_0:{}, aupr_1:{}, fpr95:{}'.format(auroc_ATT, aupr_0_ATT, aupr_1_ATT,
        #                                                                          fprs_ATT[2]))

        return embeds, logits, test_acc, auroc_ENT, aupr_0_ENT, aupr_1_ENT, fprs_ENT

    #train ood filter
    def train2(model, optimizer, iter):
        max_iters = 5
        weights = torch.ones(len(ID_classes)+1)
        
        # # Calculate the exponentially decayed value for weights[-1]
        # initial_value = 0.1  # cora: 0.2, amazon-cs: 0.01
        # decay_rate = 2  # Lambda: controls how fast the weight decreases
        # decayed_value = initial_value * math.exp(-decay_rate * iter / max_iters)
        # weights[-1] = decayed_value
        
        weights[-1] = 0.2 # cora: 0.1
        weights = weights.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        xent = nn.CrossEntropyLoss(weight = weights)
        
        ent_loss_func = EntropyLoss(reduction=False)
        best_t = args.epochs - 1
        best_metric = 0
        patience = 0
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            loss = torch.zeros(1).to(device)
            embeds, logits = model(data)
            # sup_loss = xent(logits[data.train_mask], data.y[data.train_mask])
            # sup_loss = xent(logits[data.detection_mask_train], data.joint_y_train)
            sup_loss = xent(logits[data.detection_mask_train], data.joint_y_all[data.detection_mask_train])
            loss += sup_loss
            loss.backward()
            optimizer.step()
            # validate
            if epoch % 10 == 0:
                model.eval()
                embeds, logits = model(data)
                preds = logits.argmax(axis=1).detach()
                # val_acc = get_acc(data.y, preds, data.val_mask)
                n_correct = (data.joint_y_val == preds[data.detection_mask_val]).sum()
                n_total = data.detection_mask_val.sum()

                val_acc = n_correct / n_total
                ent_loss = ent_loss_func(logits).detach().cpu()
                auroc, _, _, _ = get_ood_performance(detection_y_val, ent_loss, detection_mask_val)

                current_metric = val_acc
                if  current_metric > best_metric:
                    best_t = epoch
                    patience = 0
                    best_metric = current_metric
                    torch.save(model.state_dict(), 'best_GNN2.pkl')
                else:
                    patience += 1
                    if patience > 20:
                        break

        return best_metric, best_t
    
    #evaluate ood filter
    def evaluate2(model):
        with torch.no_grad():
            model.load_state_dict(torch.load('best_GNN2.pkl'))
            model.eval()
            # classification
            embeds, logits = model(data)
            preds = logits.argmax(axis=1).detach()
            
            n_correct = (data.joint_y_test == preds[data.detection_mask_test]).sum()
            n_total = data.detection_mask_test.sum()

            test_acc = n_correct / n_total

        return preds

    model, model2, opt, opt2 = init_oodgat()
    k = len(ID_classes) * args.splits[0]
    left_ID_indices = left_idx_all
    iter_num = 5
    adj = to_dense_adj(data.edge_index).squeeze(0)
    # adj = adj.to_sparse()
    budget_ad = len(args.ID_classes) * (n_samples_per_class-n_samples_init)
    budget_ad = int(budget_ad/5)
    idx_train = torch.where(data.train_mask)[0].cuda()
    idx_cand_an = left_idx_all
    cluster_num = 48
    
    for iter in range(iter_num + 1):
             
        #filter to train k+1 classifier
        train2(model2, opt2, iter)
        preds = evaluate2(model2)
        
        #train ID classifier
        best_metric, best_t = train_oodgat(model, opt)
        embeds, prob_nc, test_acc, auroc_ENT, aupr_0_ENT, aupr_1_ENT, fprs_ENT = evaluate_oodgat(model)
        num_id = idx_train.shape[0]
        num_total = data.detection_mask_train.sum()
        print('Number of ID nodes used for training: {}!!'.format(num_id))
        print('Total number of nodes used: {}!!'.format(num_total))
        precision = num_id/num_total
        
        #idx_cand_an able to be annotated(now unlabeled)
        preds_idx_ID = torch.nonzero(torch.ne(preds[idx_cand_an], len(args.ID_classes))).cpu()
        idx_cand_an = torch.tensor(idx_cand_an)
        preds_idx_ID = idx_cand_an[preds_idx_ID.squeeze()]
        n_correct = (data.joint_y_all[idx_cand_an] == preds[idx_cand_an]).sum()
        n_total = preds[idx_cand_an].shape[0]
        t_acc = n_correct / n_total
        joint_y_gt = data.joint_y_all[preds_idx_ID]
        mask_all = torch.eq(joint_y_gt, len(args.ID_classes))
        
        # Node Selection
        idx_selected = query_medoids_spec_nent(adj, embeds, prob_nc, budget_ad, preds_idx_ID, cluster_num)
        idx_selected = torch.tensor(idx_selected).cuda()
    
        # split ID and OOD nodes from selected nodes
        joint_y_selected = data.joint_y_all[idx_selected]
        mask = torch.eq(joint_y_selected, len(args.ID_classes))
        idx_selected_id = idx_selected[~mask.cuda()]
        idx_selected_ood = idx_selected[mask.cuda()]
        
        # Update state
        idx_train = torch.cat((idx_train, idx_selected_id))
        idx_selected = idx_selected.cpu().tolist()  # Convert tensor to list
        idx_cand_an = idx_cand_an.cpu().numpy().tolist()
        idx_cand_an = list(set(idx_cand_an)-set(idx_selected))
        
        for i in range(len(data.y)):
            # if i in idx_train:
            if i in idx_selected_id:
                data.train_mask[i] = 1
                data.detection_mask_train[i] = 1
            if i in idx_selected_ood:
                data.detection_mask_train[i] = 1
        
                
    return test_acc, auroc_ENT, aupr_0_ENT, aupr_1_ENT, fprs_ENT, precision.detach().cpu().numpy()
    

def local_ent_loss(logits, att, n_id_classes, m=0.5):
    att_norm = F.sigmoid(torch.hstack([att[0], att[1]]).mean(axis=1)).detach()  # n-dim
    mask = torch.ge(att_norm - m, 0)
    ce_uni = CE_uniform(n_id_classes, reduction=False)
    ce = ce_uni(logits)  # N-dim
    if mask.sum() > 0:
        loss = ce[mask].mean()
    else:
        loss = 0

    return loss


def set_random_seed(seed):
    np.random.seed(seed)  # Set the seed for NumPy
    random.seed(seed)     # Set the seed for the built-in random module
    torch.manual_seed(seed)  # Set the seed for PyTorch (CPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KDD22: Learning on Graphs with Out-of-Distribution nodes')
    # parser.add_argument('--dataset', default="Planetoid_Cora", type=str)
    # parser.add_argument('--ID_classes', default=[4, 2, 5, 6], type=list)
    # parser.add_argument('--dataset', default="Amazon_computers", type=str) 
    # parser.add_argument('--ID_classes', default=[1, 2, 6, 7, 8], type=list) 
    # parser.add_argument('--dataset', default="Amazon_photo", type=str) 
    # parser.add_argument('--ID_classes', default=[0, 2, 3, 4, 5], type=list)
    parser.add_argument('--dataset', default="LastFMAsia", type=str) # default:
    parser.add_argument('--ID_classes', default=[0, 6, 7, 8, 11, 13, 14, 15, 16], type=list)
    parser.add_argument('--splits', default=[5, 15, 10, 500], type=list)
    parser.add_argument('--continuous', default=False, type=bool)
    parser.add_argument('--model_name', default='OODGAT', type=str)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--hidden_dim', default=32, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--drop_prob', default=0.5, type=float)
    parser.add_argument('--drop_input', default=0.5, type=float)
    parser.add_argument('--drop_edge', default=0.6, type=float)
    parser.add_argument('--heads', default=4, type=int)
    parser.add_argument('--w_consistent', default=2.0, type=float)
    parser.add_argument('--w_ent', default=0.05, type=float)
    parser.add_argument('--w_discrepancy', default=5e-3, type=float)
    parser.add_argument('--margin', default=0.6, type=float)
    parser.add_argument('--random_seed_data', default=123, type=int)
    parser.add_argument('--random_seed_model', default=456, type=int)


    args = parser.parse_args()
    # train(args)
    n_runs = 10
    auroc_ENT_list, aupr_0_ENT_list, aupr_1_ENT_list, fprs_ENT_list, test_acc_list, precision_list = [], [], [], [], [], []
    for i in range(n_runs):
        
        random_seed = args.random_seed_data + i  # Change seed for each iteration
        set_random_seed(random_seed)
        
        test_acc, auroc_ENT, aupr_0_ENT, aupr_1_ENT, fprs_ENT, precision = train(args)
        test_acc_list.append(test_acc)
        auroc_ENT_list.append(auroc_ENT)
        aupr_0_ENT_list.append(aupr_0_ENT)
        aupr_1_ENT_list.append(aupr_1_ENT)
        fprs_ENT_list.append(fprs_ENT)
        precision_list.append(precision)
        
    print('Final Average Detection via ENT: test_acc:{}, auroc:{}, aupr_0: {}, aupr_1: {}, fpr95:{}, selection precision:{}.'.format(torch.stack(test_acc_list).mean(), np.mean(auroc_ENT_list), np.mean(aupr_0_ENT_list), np.mean(aupr_1_ENT_list), np.mean(fprs_ENT_list), np.mean(precision_list)))



