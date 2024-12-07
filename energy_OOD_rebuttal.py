import sys
import argparse
import torch
from scipy.stats import entropy
import torch.nn.functional as F
import torch.nn as nn
# pyg imports
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch_geometric.datasets import Planetoid, WikiCS, Amazon, Coauthor, LastFMAsia, NELL, CitationFull, GNNBenchmarkDataset
from torch_geometric.transforms import NormalizeFeatures, Compose
from torch_geometric.utils import homophily
from basic_gnns import MLP, GCNNet, GATNet, GATv2Net, SAGENet, GCNDetector, GATDetector
from OOD.utils import EntropyLoss, seed_torch
from metrics import get_acc, get_ood_performance, get_f1_score
from data_process import generate_masks_lego

from sampling_methods import *
from torch_geometric.utils import to_dense_adj

import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice.")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

import networkx as nx
from torch_geometric.utils import to_networkx

from data_utils import get_measures2
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree

def train(args):

    dataset_str = args.dataset.split('_')[0]
    # transforms
    trans = []
    continuous = args.continuous
    if not continuous:
        trans.append(NormalizeFeatures())

    trans = Compose(trans)

    if dataset_str == 'wiki-CS':
        dataset = WikiCS(root='pyg_data/wiki-CS', transform=trans)
    elif dataset_str == 'LastFMAsia':
        dataset = LastFMAsia(root='pyg_data/LastFMAsia', transform=trans)
    elif dataset_str == 'NELL':
        dataset = NELL(root='pyg_data/NELL', transform=trans)
    elif dataset_str == 'Amazon':
        dataset_name = args.dataset.split('_')[1]
        dataset = Amazon(root='pyg_data/Amazon', name=dataset_name, transform=trans)
    elif dataset_str == 'Coauthor':
        dataset_name = args.dataset.split('_')[1]
        dataset = Coauthor(root='pyg_data/Coauthor', name=dataset_name, transform=trans)
    elif dataset_str == 'CitationFull':
        dataset_name = args.dataset.split('_')[1]
        dataset = CitationFull(root='pyg_data/CitationFull', name=dataset_name, transform=trans)
    elif dataset_str == 'GNNBenchmark':
        dataset_name = args.dataset.split('_')[1]
        dataset = GNNBenchmarkDataset(root='pyg_data/GNNBenchmark', name=dataset_name, transform=trans)
    elif dataset_str == 'Planetoid':
        dataset_name = args.dataset.split('_')[1]
        dataset = Planetoid(root='pyg_data/Planetoid', name=dataset_name, transform=trans)
    else:
        print('unkwown dataset.')
        sys.exit()

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

    
    # Convert the PyTorch Geometric graph to a NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Graph centrality calculation
    def centralissimo(G):
        centralities = []
        centralities.append(nx.pagerank(G))  # Calculate PageRank centrality
        L = len(centralities[0])
        Nc = len(centralities)
        cenarray = np.zeros((Nc, L))
        for i in range(Nc):
            cenarray[i][list(centralities[i].keys())] = list(centralities[i].values())
        normcen = (cenarray.astype(float) - np.min(cenarray, axis=1)[:, None]) / \
                (np.max(cenarray, axis=1) - np.min(cenarray, axis=1))[:, None]
        return normcen

    # dataset parameters
    ID_classes = args.ID_classes
    splits = args.splits
    n_samples_init = splits[0]
    n_samples_per_class = splits[1]
    val_size_per_class = splits[2]
    test_size = splits[3]
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
    def init():
        # init model
        # seed_torch(args.random_seed_model)
        in_dim = data.x.shape[1]
        out_dim = len(ID_classes)
        if args.model_name == 'MLP':
            model = MLP(in_dim, args.hidden_dim, out_dim, args.drop_prob).to(device)
        elif args.model_name == 'GCN':
            model = GCNNet(in_dim, args.hidden_dim, out_dim, args.drop_prob, bias=True).to(device)
            model2 = GCNDetector(in_dim, args.hidden_dim, out_dim, args.drop_prob, bias=True).to(device)
        elif args.model_name == 'GAT':
            model = GATNet(in_dim, args.hidden_dim, out_dim, args.heads, args.drop_edge, args.drop_prob, bias=True).to(device)
            model2 = GATDetector(in_dim, args.hidden_dim, out_dim, args.heads, args.drop_edge, args.drop_prob, bias=True).to(device)
        elif args.model_name == 'GATv2':
            model = GATv2Net(in_dim, args.hidden_dim, out_dim, args.heads, args.drop_edge, True, args.drop_prob, bias=True).to(device)
        elif args.model_name == 'SAGE':
            model = SAGENet(in_dim, args.hidden_dim, out_dim, args.drop_prob, bias=True).to(device)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        return model, model2, optimizer, optimizer2

    def propagation(e, edge_index, prop_layers=1, alpha=0.5):
        '''energy belief propagation, return the energy after propagation'''
        e = e.unsqueeze(1)
        N = e.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        for _ in range(prop_layers):
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        return e.squeeze(1)
    
    def train(model, optimizer):
        # train
        xent = nn.CrossEntropyLoss()
        ent_loss_func = EntropyLoss(reduction=False)
        best_t = args.epochs - 1
        best_metric = 0
        patience = 0
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            loss = torch.zeros(1).to(device)
            embeds, logits = model(data)
            sup_loss = xent(logits[data.train_mask], data.y[data.train_mask])
            loss += sup_loss
            
            # OOD exposure
            neg_energy_all = args.T * torch.logsumexp(logits / args.T, dim=-1)
            neg_energy = args.T * torch.logsumexp(logits[data.detection_mask_train] / args.T, dim=-1)
            # neg_energy = propagation(neg_energy_all, data.edge_index, args.K, args.alpha)
            detection_y_train = data.joint_y_all[data.detection_mask_train] 
            length = len(args.ID_classes)
            energy_in = - neg_energy[torch.where(detection_y_train!=length)].cpu().detach()
            energy_out = - neg_energy[torch.where(detection_y_train==length)].cpu().detach()
            
            if energy_in.shape[0] != energy_out.shape[0]:
                min_n = min(energy_in.shape[0], energy_out.shape[0])
                energy_in = energy_in[:min_n]
                energy_out = energy_out[:min_n]

            # compute regularization loss
            reg_loss = torch.mean(F.relu(energy_in - args.m_in) ** 2 + F.relu(args.m_out - energy_out) ** 2)

            loss = sup_loss + args.lamda * reg_loss
            
            loss.backward()
            optimizer.step()
            # validate
            if epoch % 10 == 0:
                model.eval()
                embeds, logits = model(data)
                preds = logits.argmax(axis=1).detach()
                val_acc = get_acc(data.y, preds, data.val_mask)

                neg_energy_all = args.T * torch.logsumexp(logits / args.T, dim=-1)
                neg_energy = args.T * torch.logsumexp(logits[data.detection_mask_val] / args.T, dim=-1)
                # neg_energy = propagation(neg_energy_all, data.edge_index, args.K, args.alpha)
                
                labels = torch.tensor(detection_y_val, dtype=torch.int32)
                energy_in = neg_energy[torch.where(labels==0)].cpu().detach()
                energy_out = neg_energy[torch.where(labels==1)].cpu().detach()

                pos = np.array(energy_in[:]).reshape((-1, 1))
                neg = np.array(energy_out[:]).reshape((-1, 1))
                examples = np.squeeze(np.vstack((pos, neg)))
                labels = np.zeros(len(examples), dtype=np.int32)
                labels[:len(pos)] += 1
    
                auroc, aupr, fpr = get_measures2(labels, examples)
                
                current_metric = val_acc + auroc
                if  current_metric > best_metric:
                    best_t = epoch
                    patience = 0
                    best_metric = current_metric
                    torch.save(model.state_dict(), 'best_GNN.pkl')
                else:
                    patience += 1
                    if patience > 20:
                        break

        return best_metric, best_t

    def evaluate(model):
        # evaluate
        with torch.no_grad():
            model.load_state_dict(torch.load('best_GNN.pkl'))
            model.eval()
            # classification
            embeds, logits = model(data)
            preds = logits.argmax(axis=1).detach()
            test_acc = get_acc(data.y, preds, data.test_mask)
            print('test_acc:{}'.format(test_acc.item()))

            # OOD detection
            neg_energy = args.T * torch.logsumexp(logits[data.detection_mask_test] / args.T, dim=-1)
            labels = torch.tensor(detection_y_test, dtype=torch.int32)
            energy_in = neg_energy[torch.where(labels==0)].cpu().detach()
            energy_out = neg_energy[torch.where(labels==1)].cpu().detach()

            pos = np.array(energy_in[:]).reshape((-1, 1))
            neg = np.array(energy_out[:]).reshape((-1, 1))
            examples = np.squeeze(np.vstack((pos, neg)))
            labels = np.zeros(len(examples), dtype=np.int32)
            labels[:len(pos)] += 1
            
            auroc, aupr, fprs = get_measures2(labels, examples)
            print('Detection via energy: auroc:{}, aupr: {}, fpr95:{}.'.format(auroc, aupr, fprs[2]))

        return embeds, logits, test_acc, auroc, aupr, fprs

    
    def train2(model, optimizer):
        weights = torch.ones(len(ID_classes)+1)
        weights[-1] = 0.1
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
    
    model, model2, opt, opt2 = init()
    # left_ID_indices = torch.nonzero(left_ID_mask, as_tuple=True)[0]
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
        best_metric, best_t = train(model, opt)
        embeds, prob_nc, test_acc, auroc_ENT, aupr_0_ENT, fprs_ENT = evaluate(model)
        num_id = idx_train.shape[0]
        num_total = data.detection_mask_train.sum()
        print('Number of ID nodes used for training: {}!!'.format(num_id))
        print('Total number of nodes used: {}!!'.format(num_total))
        precision = num_id/num_total
        
        # # random node selection
        # idx_cand_an = np.random.permutation(left_idx_all)
        # idx_selected = idx_cand_an[:budget_ad]
        # joint_y_selected = data.joint_y_all[idx_selected]
        # mask = torch.eq(joint_y_selected, len(args.ID_classes))
        # idx_selected_id = idx_selected[~mask.cpu()]
        # idx_selected_ood = idx_selected[mask.cpu()]
        # idx_train = torch.cat((idx_train, torch.tensor(idx_selected_id).cuda()))
        # idx_selected = idx_selected.tolist()  # Convert tensor to list
        # idx_cand_an = list(set(idx_cand_an)-set(idx_selected))
        
        # Node Selection based on active learning methods
        # idx_selected = query_medoids_spec_nent(adj, embeds, prob_nc, budget_ad, idx_cand_an, cluster_num)
        # idx_selected = query_uncertainty(prob_nc, budget_ad, idx_cand_an)
        idx_selected = query_medoids(embeds, prob_nc, budget_ad, idx_cand_an, cluster_num)
        idx_selected = torch.tensor(idx_selected).cuda()
        # split ID and OOD nodes from selected nodes
        joint_y_selected = data.joint_y_all[idx_selected]
        mask = torch.eq(joint_y_selected, len(args.ID_classes))
        idx_selected_id = idx_selected[~mask.cuda()]
        idx_selected_ood = idx_selected[mask.cuda()]
        # Update state
        idx_train = torch.cat((idx_train, torch.tensor(idx_selected_id).cuda()))
        idx_selected = idx_selected.cpu().numpy().tolist()  # Convert tensor to list
        idx_cand_an = list(set(idx_cand_an)-set(idx_selected))

        
        for i in range(len(data.y)):
            # if i in idx_train:
            if i in idx_selected_id:
                data.train_mask[i] = 1
                data.detection_mask_train[i] = 1
            if i in idx_selected_ood:
                data.detection_mask_train[i] = 1
                
    return test_acc, auroc_ENT, aupr_0_ENT, fprs_ENT, precision.detach().cpu().numpy()

def set_random_seed(seed):
    np.random.seed(seed)  # Set the seed for NumPy
    random.seed(seed)     # Set the seed for the built-in random module
    torch.manual_seed(seed)  # Set the seed for PyTorch (CPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline: some commonly adopted GNNs.')

    parser.add_argument('--dataset', default="Planetoid_Cora", type=str) # default: "Planetoid_Cora" splits: [1,10,500]
    parser.add_argument('--ID_classes', default=[4, 2, 5, 6], type=list) # default: [4, 2, 5, 6]
    # parser.add_argument('--dataset', default="Amazon_computers", type=str) # default: "Planetoid_Cora"
    # parser.add_argument('--ID_classes', default=[1, 2, 6, 7, 8], type=list) # default: [4, 2, 5, 6]
    # parser.add_argument('--dataset', default="Amazon_photo", type=str) # default: splits: [2,10,500]
    # parser.add_argument('--ID_classes', default=[0, 2, 3, 4, 5], type=list) # default: [4, 2, 5, 6]
    # parser.add_argument('--dataset', default="wiki-CS", type=str) # default:
    # parser.add_argument('--ID_classes', default=[1, 3, 6, 7, 8, 9], type=list)
    # parser.add_argument('--dataset', default="LastFMAsia", type=str) # default:
    # parser.add_argument('--ID_classes', default=[0, 6, 7, 8, 11, 13, 14, 15, 16], type=list)
    # parser.add_argument('--splits', default=[5, 10, 500], type=list)
    parser.add_argument('--splits', default=[5, 15, 10, 500], type=list)
    parser.add_argument('--continuous', default=False, type=bool)
    parser.add_argument('--model_name', default='GCN', type=str)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--hidden_dim', default=32, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--drop_prob', default=0.5, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--drop_edge', default=0.6, type=float)
    parser.add_argument('--heads', default=4, type=int)
    parser.add_argument('--random_seed_data', default=123, type=int)
    parser.add_argument('--random_seed_model', default=456, type=int)

    parser.add_argument('--T', type=float, default=1.0, help='temperature for Softmax')
    parser.add_argument('--use_reg', action='store_false', help='whether to use energy regularization loss')
    parser.add_argument('--lamda', type=float, default=0.01, help='weight for regularization')
    parser.add_argument('--m_in', type=float, default=-5, help='upper bound for in-distribution energy')
    parser.add_argument('--m_out', type=float, default=-1, help='lower bound for in-distribution energy')
    parser.add_argument('--use_prop', action='store_false', help='whether to use energy belief propagation')
    parser.add_argument('--K', type=int, default=2, help='number of layers for energy belief propagation')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual connection in propagation')

    args = parser.parse_args()
    n_runs = 10
    auroc_ENT_list, aupr_0_ENT_list, fprs_ENT_list, test_acc_list, precision_list = [], [], [], [], []
    for i in range(n_runs):
        random_seed = args.random_seed_data + i  # Change seed for each iteration
        set_random_seed(random_seed)
        
        test_acc, auroc_ENT, aupr_0_ENT, fprs_ENT, precision = train(args)
        test_acc_list.append(test_acc)
        auroc_ENT_list.append(auroc_ENT)
        aupr_0_ENT_list.append(aupr_0_ENT)
        fprs_ENT_list.append(fprs_ENT)
        precision_list.append(precision)
        
    print('Final Average Detection via energy: test_acc:{}, auroc:{}, aupr_0: {}, fpr95:{}, selection precision:{}.'.format(torch.stack(test_acc_list).mean(), np.mean(auroc_ENT_list), np.mean(aupr_0_ENT_list), np.mean(fprs_ENT_list), np.mean(precision_list)))

