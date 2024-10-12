import torch
import numpy as np
        

def generate_masks_lego(data, n_samples_init, ID_classes, n_samples_per_class, val_size_per_class, test_size, random_seed=123):
    # seed = np.random.get_state()
    # np.random.seed(random_seed)
    
    classes = set(data.y.tolist())
    n_classes = len(classes)
    ID_classes = set(ID_classes)
    assert len(ID_classes) < n_classes
    OOD_classes = classes - ID_classes

    n_nodes = len(data.y)
    ID_mask, OOD_mask = torch.zeros(n_nodes), torch.zeros(n_nodes)
    for i in range(n_nodes):
        if data.y[i].item() in ID_classes:
            ID_mask[i] = 1
        else:
            OOD_mask[i] = 1
    ID_mask, OOD_mask = ID_mask.bool(), OOD_mask.bool()
    assert n_samples_per_class*len(ID_classes) + val_size_per_class*len(ID_classes) + test_size <= ID_mask.sum()
    assert n_samples_per_class*len(ID_classes) + val_size_per_class*len(ID_classes) + test_size <= OOD_mask.sum()

    # ID part
    ID_idx = torch.nonzero(ID_mask).squeeze().tolist()
    train_idx = []
    val_idx = []
    for k in ID_classes:
        k_idxs = torch.nonzero(data.y==k).squeeze().tolist()
        samples_to_take = val_size_per_class
        samples = np.random.choice(k_idxs, samples_to_take, False)
        # train_idx.extend(samples[:n_samples_init])
        val_idx.extend(samples)
        
    left_idx1 = list(set(ID_idx)-set(val_idx))
    left_idx1 = np.random.permutation(left_idx1)
    test_idx = left_idx1[:test_size]
    left_idx_ID = list(set(ID_idx)-set(test_idx)-set(val_idx))
    
    train_mask, val_mask, test_mask = torch.zeros(n_nodes), torch.zeros(n_nodes), torch.zeros(n_nodes)
    left_ID_mask = torch.zeros(n_nodes)
    
    for i in range(n_nodes):
        if i in left_idx_ID:
            left_ID_mask[i] = 1
        
        if i in train_idx:
            train_mask[i] = 1
        elif i in val_idx:
            val_mask[i] = 1
        elif i in test_idx:
            test_mask[i] = 1
    train_mask_ID, val_mask_ID, test_mask_ID, left_ID_mask = train_mask.bool(), val_mask.bool(), test_mask.bool(), left_ID_mask.bool()

    # OOD part
    OOD_idx = torch.nonzero(OOD_mask).squeeze().tolist()
    OOD_idx = np.random.permutation(OOD_idx)
    train_idx = OOD_idx[: train_mask_ID.sum()]
    val_idx = OOD_idx[train_mask_ID.sum(): train_mask_ID.sum()+val_mask_ID.sum()]
    test_idx = OOD_idx[train_mask_ID.sum()+val_mask_ID.sum(): train_mask_ID.sum()+val_mask_ID.sum()+test_size]
    train_mask, val_mask, test_mask = torch.zeros(n_nodes), torch.zeros(n_nodes), torch.zeros(n_nodes)
    left_OOD_mask = torch.zeros(n_nodes)
    left_idx_OOD = list(set(OOD_idx)-set(test_idx)-set(val_idx))
    
    for i in range(n_nodes):
        if i in left_idx_OOD:
            left_OOD_mask[i] = 1
        if i in train_idx:
            train_mask[i] = 1
        elif i in val_idx:
            val_mask[i] = 1
        elif i in test_idx:
            test_mask[i] = 1
    
    train_mask_OOD, val_mask_OOD, test_mask_OOD, left_OOD_mask = train_mask.bool(), val_mask.bool(), test_mask.bool(), left_OOD_mask.bool()

    # generate random training nodes 
    left_idx_all = list(set(left_idx_ID)|set(left_idx_OOD))
    left_idx_all = np.random.permutation(left_idx_all)
    selected_idx = left_idx_all[:n_samples_init*len(ID_classes)]
    selected_idx_ID, selected_idx_OOD = [], []
    for i in range(len(selected_idx)):
        if data.y[selected_idx[i]].item() in ID_classes:
            selected_idx_ID.append(selected_idx[i])
        else:
            selected_idx_OOD.append(selected_idx[i])
    
    for i in range(n_nodes):
        if i in selected_idx_ID:
            train_mask_ID[i] = 1
        elif i in selected_idx_OOD:
            train_mask_OOD[i] = 1
    
    left_idx_all = list(set(left_idx_all)-set(selected_idx))
    
    # detection
    detection_mask_val = val_mask_ID | val_mask_OOD
    detection_y_val = data.y[detection_mask_val]
    detection_y_val = [y.item() in OOD_classes for y in detection_y_val]
    detection_mask_test = test_mask_ID | test_mask_OOD
    detection_y_test = data.y[detection_mask_test]
    detection_y_test = [y.item() in OOD_classes for y in detection_y_test]

    # re-map labels
    label_map = {y : i for i, y in enumerate(ID_classes)}

    # ID classification y
    ID_y = data.y.clone()
    for i in range(n_nodes):
        ID_y[i] = label_map.get(data.y[i].item(), 0)

    # joint clf y
    joint_y_val = data.y[detection_mask_val]
    joint_y_val = [label_map.get(y.item(), len(ID_classes)) for y in joint_y_val]
    joint_y_test = data.y[detection_mask_test]
    joint_y_test = [label_map.get(y.item(), len(ID_classes)) for y in joint_y_test]

    # for training detector
    detection_mask_train = train_mask_ID | train_mask_OOD
    joint_y_train = data.y[detection_mask_train]
    joint_y_train = [label_map.get(y.item(), len(ID_classes)) for y in joint_y_train]

    joint_y_all = data.y
    joint_y_all = [label_map.get(y.item(), len(ID_classes)) for y in joint_y_all]
    
    # reset seed
    # np.random.set_state(seed)

    return ID_mask, OOD_mask, train_mask_ID, val_mask_ID, test_mask_ID, train_mask_OOD, val_mask_OOD, test_mask_OOD, \
        detection_mask_val, detection_y_val, joint_y_val, detection_mask_test, detection_y_test, joint_y_test, \
        ID_y, left_idx_all, detection_mask_train, joint_y_train, joint_y_all

#
def normalize_feature(x):
    x_new = x.clone()
    sum = x_new.sum(axis=1, keepdims=True)
    x_new = x_new / sum
    x_new[np.isinf(x_new)] = 0

    return x_new

