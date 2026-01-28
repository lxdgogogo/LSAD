import torch




def train_test(label: torch.Tensor, few_shot_num=5, device="cuda"):
    ones_idx = torch.nonzero(label == 1)  
    zeros_idx = torch.nonzero(label == 0)  
    idx_all = torch.arange(label.shape[0], device=device)
    selected_ones_idx = ones_idx[torch.randperm(ones_idx.size(0))[:few_shot_num]] 
    selected_zeros_idx = zeros_idx[torch.randperm(zeros_idx.size(0))[:few_shot_num]]  
    train_idx = torch.cat([selected_ones_idx, selected_zeros_idx])
    test_idx = idx_all[~torch.isin(label, train_idx)]
    train_set = label[train_idx]
    test_set = label[test_idx]
    return train_idx, train_set, test_idx, test_set
