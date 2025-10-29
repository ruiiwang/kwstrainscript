import os
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model.crnn_model import CnnRnnModel1Channel

# 模型配置
# 输出2个神经元，分别代表样本0和样本1的概率
config = {
    "in_c": 16,
    "conv": [{"out_c": 16, "k": 8, "s": 2, "p": 1, "dropout": 0.0},
            {"out_c": 32, "k": 4, "s": 2, "p": 1, "dropout": 0.0}],
    "rnn": {"dim": 32, "layers": 1, "dropout": 0.9, "bidirectional": True},
    "fc_out": 2  # 修改为2个输出神经元：[样本0概率, 样本1概率]
}

def log_message(message, log_file):
    print(message)
    if log_file:
        log_file.write(message + '\n')
        log_file.flush()

# 加载所有pkl文件，现在只加载正样本
def load_all_pkls(pkl_dirs, pos_label=1):
    features = []
    labels = []
    for pkl_dir in pkl_dirs:
        if not os.path.exists(pkl_dir):
            log_message(f"警告: 目录 {pkl_dir} 不存在，跳过加载。", None)
            continue
        for pkl_file in os.listdir(pkl_dir):
            if pkl_file.endswith('.pkl'):
                with open(os.path.join(pkl_dir, pkl_file), 'rb') as f:
                    data = pickle.load(f)
                    feat = data[0]
                    original_labels = data[1]  # pkl内原始标签（int）
                    # 二值化：pos_label 为正样本，其余为负样本
                    binary_labels = (original_labels == pos_label).to(torch.float)
                    features.append(feat)
                    labels.append(binary_labels)
    if not features:
        raise ValueError("没有找到任何pkl文件，请检查目录设置。")
    return torch.cat(features), torch.cat(labels)

def load_pkls_with_sampling_bin(pkl_dirs, percents=None, pos_label=1):
    """
    二值化标签的多目录采样加载：pos_label 视为正样本，其余为负样本(0)。
    percents 支持 [0~1] 或 [0~100]。
    """
    features_all = []
    labels_all = []
    if percents is None:
        percents = [1.0] * len(pkl_dirs)

    for i, pkl_dir in enumerate(pkl_dirs):
        feats_dir = []
        labels_dir = []
        if not os.path.exists(pkl_dir):
            log_message(f"警告: 目录 {pkl_dir} 不存在，跳过加载。", None)
            continue
        for pkl_file in os.listdir(pkl_dir):
            if pkl_file.endswith('.pkl'):
                with open(os.path.join(pkl_dir, pkl_file), 'rb') as f:
                    data = pickle.load(f)
                    feat = data[0]
                    original_labels = data[1]
                    binary_labels = (original_labels == pos_label).to(torch.float)
                    feats_dir.append(feat)
                    labels_dir.append(binary_labels)
        if not feats_dir:
            continue

        dir_features = torch.cat(feats_dir)
        dir_labels = torch.cat(labels_dir)

        frac = percents[i]
        if frac > 1.0:
            frac = frac / 100.0
        frac = max(0.0, min(1.0, frac))

        take_n = int(frac * dir_features.size(0))
        if take_n == 0 and frac > 0.0:
            take_n = 1
        if take_n > 0:
            idx = torch.randperm(dir_features.size(0))[:take_n]
            dir_features = dir_features[idx]
            dir_labels = dir_labels[idx]
            features_all.append(dir_features)
            labels_all.append(dir_labels)

    if not features_all:
        raise ValueError("没有找到任何可用的样本，请检查目录及采样比例设置。")
    return torch.cat(features_all), torch.cat(labels_all)

def train_model(model, features, labels, epochs=10, batch_size=32, folder='checkpoint', resume_checkpoint=None,
                fine_tune=False, learning_rate=None, loss_type='bce', early_stopping_metric='val_loss',
                patience=5, threshold=0.5, freeze_frontend=True):
    # 划分数据集: 60%训练集, 20%验证集, 20%测试集
    dataset_size = len(features)
    train_size = int(0.6 * dataset_size)
    val_size = int(0.2 * dataset_size)
    
    # 随机划分
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # 创建数据集
    train_dataset = TensorDataset(features[train_indices], labels[train_indices])
    val_dataset = TensorDataset(features[val_indices], labels[val_indices])
    test_dataset = TensorDataset(features[test_indices], labels[test_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 设备（GPU 优先）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 计算正样本权重（用于 weighted_bce）
    train_targets = labels[train_indices]
    pos_count = float(train_targets.sum().item())
    neg_count = float(train_targets.numel()) - pos_count
    pos_weight_value = (neg_count / (pos_count + 1e-9)) if pos_count > 0 else 1.0
    pos_weight_tensor = torch.tensor(pos_weight_value, device=device)

    # 损失函数：只针对“样本1的logit”计算BCE，保持与train_one_class一致
    class FocalBCE(nn.Module):
        def __init__(self, alpha=None, gamma=2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
        def forward(self, logits_pos, targets):
            probs_pos = torch.sigmoid(logits_pos)
            ce = nn.functional.binary_cross_entropy_with_logits(
                logits_pos, targets, pos_weight=self.alpha, reduction='none'
            )
            p_t = probs_pos * targets + (1 - probs_pos) * (1 - targets)
            loss = ((1 - p_t) ** self.gamma) * ce
            return loss.mean()

    if loss_type == 'weighted_bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    elif loss_type == 'focal_bce':
        criterion = FocalBCE(alpha=pos_weight_tensor, gamma=2.0)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # 适度增大 RNN dropout 以增强微调鲁棒性
    if fine_tune and hasattr(model, 'rnn'):
        try:
            model.rnn.dropout = max(getattr(model.rnn, 'dropout', 0.0), 0.4)
        except Exception:
            pass

    if not os.path.exists(folder):
        os.makedirs(folder)
    log_file = open(os.path.join(folder, 'training_log.txt'), 'a')
    start_epoch = 0
    best_metric = None
    metric_mode = 'min' if early_stopping_metric in ('val_loss', 'far') else 'max'
    no_improve_epochs = 0

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        log_message(f'Continue training from {resume_checkpoint} ...', log_file)
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if not fine_tune and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) if not fine_tune else 0
        last_loss = checkpoint.get('loss', None)
        if last_loss is not None:
            log_message(f'Continue training from Epoch {start_epoch + 1} , last train loss: {last_loss:.4f}', log_file)
        log_message(f'Device: {device}', log_file)
        log_message(f'Total samples: {len(features)}', log_file)
        log_message(f'batch_size: {batch_size}', log_file)
        log_message(f'Epochs: {epochs}', log_file)
        log_message(f'Fine-tune: {fine_tune}, LR: {learning_rate}, Loss: {loss_type}', log_file)
    else:
        log_message('Start!', log_file)
        log_message(f'Device: {device}', log_file)
        log_message(f'Total samples: {len(features)}', log_file)
        log_message(f'batch_size: {batch_size}', log_file)
        log_message(f'Epochs: {epochs}', log_file)
        log_message(f'Fine-tune: {fine_tune}, LR: {learning_rate}, Loss: {loss_type}', log_file)
        log_message(f'Computed pos_weight: {pos_weight_value:.4f}', log_file)

    def evaluate(model, loader):
        model.eval()
        total = 0
        correct = 0
        total_loss = 0.0
        tp = fp = tn = fn = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)  # [batch_size, 2]
                logit_pos = outputs[:, 1]  # 只用样本1的logit
                batch_loss = criterion(logit_pos, targets)
                total_loss += batch_loss.item()

                prob1 = torch.sigmoid(logit_pos)
                prob0 = 1.0 - prob1  # 衍生出的样本0概率
                preds = (prob1 >= threshold).float()

                total += targets.size(0)
                correct += (preds == targets).sum().item()
                tp += ((preds == 1) & (targets == 1)).sum().item()
                fp += ((preds == 1) & (targets == 0)).sum().item()
                tn += ((preds == 0) & (targets == 0)).sum().item()
                fn += ((preds == 0) & (targets == 1)).sum().item()

        val_loss = total_loss / max(len(loader), 1)
        acc = 100.0 * correct / max(total, 1)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        far = fp / (fp + tn + 1e-9)
        return val_loss, acc, f1, far

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        prob_sum = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # [batch_size, 2]
            logit_pos = outputs[:, 1]  # 只用样本1的logit
            loss = criterion(logit_pos, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            prob1 = torch.sigmoid(logit_pos)
            prob0 = 1.0 - prob1  # 衍生出的样本0概率
            preds = (prob1 >= threshold).float()
            total += targets.size(0)
            correct += (preds == targets).sum().item()
            prob_sum += prob1.sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / max(total, 1)
        avg_pos_prob = prob_sum / max(total, 1)

        val_loss, val_acc, val_f1, val_far = evaluate(model, val_loader)
        log_message(
            f'Epoch {epoch+1}/{epochs}, '
            f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, AvgClass1Prob: {avg_pos_prob:.4f}, '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}, Val FAR: {val_far:.4f}',
            log_file
        )

        checkpoint_dict = {
            "epoch": epoch+1,
            "loss": epoch_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        torch.save(checkpoint_dict, f'{folder}/epoch{epoch+1}.pth')

        # 早停/最佳模型
        current_metric = {
            'val_loss': val_loss,
            'acc': val_acc,
            'f1': val_f1,
            'far': val_far
        }.get(early_stopping_metric, val_loss)

        improved = (best_metric is None) or (
            (metric_mode == 'min' and current_metric < best_metric) or
            (metric_mode == 'max' and current_metric > best_metric)
        )
        if improved:
            best_metric = current_metric
            torch.save(model.state_dict(), f'{folder}/crnn_model_best.pth')
            no_improve_epochs = 0
            log_message(f'New best {early_stopping_metric}: {best_metric:.4f}', log_file)
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            log_message(f'Early stopping triggered. Best {early_stopping_metric}: {best_metric:.4f}', log_file)
            break

    log_message(
        f'Best model saved. Best {early_stopping_metric}: {best_metric:.4f}' if best_metric is not None
        else 'Finish without best metric computed.',
        log_file
    )
    log_file.close()

if __name__ == "__main__":
    # 多目录（如旧数据、新数据），标签在 pkl 中，pos_label=1 为正样本
    pkl_data_dirs = ['converted_2']  # 可根据实际情况调整
    
    # 选择训练模式
    run_from_scratch = True  # True: 从零开始训练；False: 迭代微调
    
    # 采样比例设置（微调时：旧数据 n%，新数据 m%；从零开始时：均为 100%）
    sample_percent_old = 40   # 旧数据 n%
    sample_percent_new = 100  # 新数据 m%
    effective_percents = [
        100 if run_from_scratch else sample_percent_old,
        100 if run_from_scratch else sample_percent_new
    ]
    
    # 按目录比例加载数据（二值化标签）
    features, labels = load_pkls_with_sampling_bin(pkl_data_dirs, effective_percents, pos_label=1)
    
    # 创建模型
    model = CnnRnnModel1Channel(config)
    
    # 设置要恢复的检查点路径
    resume_checkpoint_path = None  # 例如: 'checkpoint_1.1/epoch10.pth'
    
    if run_from_scratch:
        # 从零开始训练（标准配置）
        train_model(
            model, features, labels,
            epochs=100,                 # 你可以按需增减
            batch_size=64,             # 32/64 较合适
            folder='checkpoint_1.1',
            resume_checkpoint=None,
            fine_tune=False,
            learning_rate=1e-3,
            loss_type='bce',           # 标准 BCE
            early_stopping_metric='val_loss',
            patience=10,
            threshold=0.5,
            freeze_frontend=False
        )
    else:
        # 迭代微调：更小学习率、冻结前端、加权 BCE、按 FAR 早停
        train_model(
            model, features, labels,
            epochs=30,                 # 微调一般 10~30 就能收敛
            batch_size=32,             # 微调阶段 16/32
            folder='checkpoint_1.1_ft',
            resume_checkpoint=resume_checkpoint_path,
            fine_tune=True,            # 启用微调模式
            learning_rate=1e-4,        # 微调更小 LR
            loss_type='weighted_bce',  # 加权 BCE（解决不平衡）
            early_stopping_metric='far',
            patience=5,
            threshold=0.5,
            freeze_frontend=True       # 冻结 conv/rnn，只训练 fc
        )