import os
import pickle
import numpy as np
import torch

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, (list, tuple)) or len(data) != 2:
        raise ValueError(f"文件格式不符合 (features, labels): {pkl_path}")
    features, labels = data
    if not isinstance(features, torch.FloatTensor):
        features = features.type(torch.float)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.int32)
    elif labels.dtype != torch.int32:
        labels = labels.type(torch.int32)
    return features, labels

def sample_indices(n, k, shuffle=True):
    k = min(k, n)
    if shuffle:
        return np.random.choice(n, size=k, replace=False)
    else:
        return np.arange(k)

def mix_two_pkls(pos_pkl, neg_pkl, pos_take, neg_take, output_pkl, seed=2025, shuffle_final=True):
    np.random.seed(seed)

    pos_feat, pos_lab = load_pkl(pos_pkl)
    neg_feat, neg_lab = load_pkl(neg_pkl)

    if pos_feat.ndim != 3 or neg_feat.ndim != 3:
        raise ValueError("特征维度不为3D (N, C, T)，请检查输入pkl。")
    if pos_feat.shape[1:] != neg_feat.shape[1:]:
        raise ValueError(f"特征维度不一致: {pos_feat.shape[1:]} vs {neg_feat.shape[1:]}")
    
    pos_idx = sample_indices(pos_feat.shape[0], pos_take, shuffle=True)
    neg_idx = sample_indices(neg_feat.shape[0], neg_take, shuffle=True)

    pos_sel_feat = pos_feat[pos_idx]
    pos_sel_lab = pos_lab[pos_idx]
    neg_sel_feat = neg_feat[neg_idx]
    neg_sel_lab = neg_lab[neg_idx]

    mixed_feat = torch.cat([pos_sel_feat, neg_sel_feat], dim=0)
    mixed_lab = torch.cat([pos_sel_lab, neg_sel_lab], dim=0)

    if shuffle_final:
        perm = torch.randperm(mixed_feat.shape[0])
        mixed_feat = mixed_feat[perm]
        mixed_lab = mixed_lab[perm]

    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump((mixed_feat, mixed_lab), f)

    uniq, cnts = torch.unique(mixed_lab, return_counts=True)
    print(f"保存完成: {output_pkl}")
    print(f"  特征形状: {mixed_feat.shape}")
    print(f"  标签形状: {mixed_lab.shape}")
    print(f"  标签分布: {dict(zip(uniq.tolist(), cnts.tolist()))}")

if __name__ == "__main__":
    # 配置：根据你的实际文件修改
    pos_pkl_path = r"/mnt/d/kwstrainscript/converted_2/HeyMemo_data.pkl"        # 正类（HeyMemo=1）
    neg_pkl_path = r"/mnt/d/kwstrainscript/converted_un/wrong_segments_2_data.pkl"   # 负类（UNKNOWN_WORD=0）
    output_pkl_path = r"/mnt/d/kwstrainscript/mixed_quant.pkl"

    # 抽取数量：各取多少样本（会自动裁剪到不超过各自总数）
    pos_take_count = 500
    neg_take_count = 4500

    mix_two_pkls(
        pos_pkl=pos_pkl_path,
        neg_pkl=neg_pkl_path,
        pos_take=pos_take_count,
        neg_take=neg_take_count,
        output_pkl=output_pkl_path,
        seed=2025,
        shuffle_final=True
    )