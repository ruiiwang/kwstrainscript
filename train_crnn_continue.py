import os
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model.crnn_model import CnnRnnModel1Channel

# 模型配置
config = {
    "in_c": 16,
    "conv": [{"out_c": 16, "k": 8, "s": 2, "p": 1, "dropout": 0.0},
             {"out_c": 32, "k": 4, "s": 2, "p": 1, "dropout": 0.0}],
    "rnn": {"dim": 32, "layers": 1, "dropout": 0.2, "bidirectional": True},
    "fc_out": 2  # 类别数
}

def log_message(message, log_file):
    print(message)
    if log_file is not None:
        log_file.write(message + '\n')
        log_file.flush()

# 加载所有pkl文件，支持从多个目录加载
def load_all_pkls(pkl_dirs):
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
                    features.append(feat)
                    labels.append(data[1].long())
    if not features:
        raise ValueError("没有找到任何pkl文件，请检查目录设置。")
    return torch.cat(features), torch.cat(labels)

def load_pkls_with_sampling(pkl_dirs, percents=None):
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
                    feats_dir.append(data[0])
                    labels_dir.append(data[1].long())
        if not feats_dir:
            continue
        dir_features = torch.cat(feats_dir)
        dir_labels = torch.cat(labels_dir)
        frac = percents[i]
        # 支持传入 0~1 或 0~100，两种格式
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
                fine_tune=False, learning_rate=None, loss_type='ce', focal_gamma=2.0,
                early_stopping_metric='val_loss', patience=5, pos_class_id=1, freeze_frontend=True):
    # 划分数据集: 60%训练集, 20%验证集, 20%测试集
    dataset_size = len(features)
    train_size = int(0.6 * dataset_size)
    val_size = int(0.2 * dataset_size)
    # test_size = dataset_size - train_size - val_size

    # 随机划分
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # 创建数据集
    train_dataset = TensorDataset(features[train_indices], labels[train_indices])
    val_dataset = TensorDataset(features[val_indices], labels[val_indices])
    test_dataset = TensorDataset(features[test_indices], labels[test_indices])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 定义损失（支持 CE / Weighted CE / Focal Loss）
    num_classes = int(model.fc.out_features) if hasattr(model, 'fc') else int(labels.max().item() + 1)
    class_weights = None
    if loss_type in ('weighted_ce', 'focal'):
        # 用训练集标签分布计算权重
        train_label_counts = torch.bincount(labels[train_indices], minlength=num_classes).float()
        class_weights = 1.0 / (train_label_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * num_classes  # 归一化

    class FocalLoss(nn.Module):
        def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
            self.reduction = reduction

        def forward(self, inputs, targets):
            logp = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
            p = torch.exp(-logp)
            loss = (1 - p) ** self.gamma * logp
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            return loss

    if loss_type == 'focal':
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma, reduction='mean')
    elif loss_type == 'weighted_ce' and class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # 冻结与学习率设置
    if fine_tune and freeze_frontend:
        # 冻结前端 conv 与 rnn，仅训练 fc
        if hasattr(model, 'conv'):
            for p in model.conv.parameters():
                p.requires_grad = False
        if hasattr(model, 'rnn'):
            for p in model.rnn.parameters():
                p.requires_grad = False

    if learning_rate is None:
        learning_rate = 1e-4 if fine_tune else 1e-3

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # 微调阶段可适度增大 RNN dropout 至 0.4（若模型支持）
    if fine_tune and hasattr(model, 'rnn'):
        try:
            model.rnn.dropout = max(getattr(model.rnn, 'dropout', 0.0), 0.4)
        except Exception:
            pass

    # 确保checkpoint目录存在
    if not os.path.exists(folder):
        os.makedirs(folder)
    # 初始化日志文件
    log_file = open(os.path.join(folder, 'training_log.txt'), 'a')
    start_epoch = 0

    # 早停与最佳模型指标
    best_metric = None
    metric_mode = 'min' if early_stopping_metric in ('val_loss', 'far') else 'max'
    no_improve_epochs = 0

    # 载入 checkpoint（微调模式不加载优化器状态，并从 epoch 0 开始）
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        log_message(f'Continue training from {resume_checkpoint} ...', log_file)
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        if not fine_tune and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) if not fine_tune else 0
        last_loss = checkpoint.get('loss', None)
        if last_loss is not None:
            log_message(f'Continue training from Epoch {start_epoch + 1} , last train loss: {last_loss:.4f}', log_file)
        # 在恢复训练分支也输出基础信息，保持与未恢复分支一致
        log_message(f'Total samples: {len(features)}', log_file)
        log_message(f'batch_size: {batch_size}', log_file)
        log_message(f'Epochs: {epochs}', log_file)
        log_message(f'Fine-tune: {fine_tune}, LR: {learning_rate}, Loss: {loss_type}', log_file)
    else:
        log_message('Start!', log_file)
        log_message(f'Total samples: {len(features)}', log_file)
        log_message(f'batch_size: {batch_size}', log_file)
        log_message(f'Epochs: {epochs}', log_file)
        log_message(f'Fine-tune: {fine_tune}, LR: {learning_rate}, Loss: {loss_type}', log_file)

    # 验证评估函数（计算 loss / Acc / F1 / FAR）
    def evaluate(model, loader):
        model.eval()
        total = 0
        correct = 0
        total_loss = 0.0
        conf = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        with torch.no_grad():
            for inputs, targets in loader:
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
                total_loss += batch_loss.item()
                _, preds = outputs.max(1)
                total += targets.size(0)
                correct += preds.eq(targets).sum().item()
                for t, p in zip(targets.view(-1), preds.view(-1)):
                    conf[t.long(), p.long()] += 1
        val_loss = total_loss / max(len(loader), 1)
        acc = 100. * correct / max(total, 1)
        # 计算针对 pos_class_id 的 F1 与 FAR
        tp = conf[pos_class_id, pos_class_id].item()
        fp = (conf[:, pos_class_id].sum() - tp).item()
        fn = (conf[pos_class_id, :].sum() - tp).item()
        tn = (conf.sum() - (tp + fp + fn)).item()
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        far = fp / (fp + tn + 1e-9)  # False Accept Rate
        return val_loss, acc, f1, far

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 训练阶段
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / max(total, 1)

        # 验证阶段
        val_loss, val_acc, val_f1, val_far = evaluate(model, val_loader)
        log_message(
            f'Epoch {epoch + 1}/{epochs}, '
            f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
            f'Val F1(pos={pos_class_id}): {val_f1:.4f}, Val FAR(pos={pos_class_id}): {val_far:.4f}',
            log_file
        )

        # 保存checkpoint
        checkpoint_dict = {
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        torch.save(checkpoint_dict, f'{folder}/epoch{epoch + 1}.pth')

        # 更新最佳模型（按选择的早停指标）
        if early_stopping_metric == 'val_loss':
            current_metric = val_loss
        elif early_stopping_metric == 'f1':
            current_metric = val_f1
        elif early_stopping_metric == 'far':
            current_metric = val_far
        else:
            current_metric = val_loss

        improved = (best_metric is None) or (
            (metric_mode == 'min' and current_metric < best_metric) or
            (metric_mode == 'max' and current_metric > best_metric)
        )
        if improved:
            best_metric = current_metric
            best_model = model.state_dict()
            torch.save(best_model, f'{folder}/crnn_model_best.pth')
            no_improve_epochs = 0
            log_message(f'New best {early_stopping_metric}: {best_metric:.4f}', log_file)
        else:
            no_improve_epochs += 1

        # 早停
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
    # 设置要加载的pkl文件目录列表（第一个为旧数据，第二个为新数据）
    pkl_data_dirs = ['converted_2', 'converted_2_ft']

    # 创建模型
    model = CnnRnnModel1Channel(config)

    # 选择训练模式
    run_from_scratch = False  # True: 从零开始训练；False: 迭代微调

    # 采样比例设置（微调时：旧数据 n%，新数据 m%；从零开始时：均为 100%）
    sample_percent_old = 40   # 旧数据 n%
    sample_percent_new = 100  # 新数据 m%
    effective_percents = [
        100 if run_from_scratch else sample_percent_old,
        100 if run_from_scratch else sample_percent_new
    ]

    # 按目录比例加载数据
    features, labels = load_pkls_with_sampling(pkl_data_dirs, effective_percents)

    # 设置要恢复的检查点路径
    resume_checkpoint_path = 'checkpoint_2.2/epoch94.pth' if not run_from_scratch else None

    if run_from_scratch:
        # 从零开始训练（标准配置）
        train_model(
            model, features, labels,
            epochs=100,              # 初次训练建议 100
            batch_size=64,           # 初次训练建议 32/64
            folder='checkpoint_2.0',
            resume_checkpoint=None,  # 从零开始不加载 checkpoint
            fine_tune=False,         # 非微调
            learning_rate=1e-3,      # 初次训练较大学习率
            loss_type='ce',          # 标准 CE
            early_stopping_metric='val_loss',
            patience=10,
            pos_class_id=1,
            freeze_frontend=False    # 不冻结前端
        )
    else:
        # 迭代训练（微调）：更小学习率、较小batch、冻结前端、Focal Loss、按F1早停
        train_model(
            model, features, labels,
            epochs=30,                 # 微调一般 10~30 就能收敛
            batch_size=32,             # 微调阶段 8/16/32
            folder='checkpoint_2.2_ft',
            resume_checkpoint=resume_checkpoint_path,
            fine_tune=True,            # 启用微调模式
            learning_rate=1e-4,        # 微调更小 LR
            loss_type='weighted_ce',         # 可选 'weighted_ce' 或 'focal'
            focal_gamma=2.0,           # Focal Loss gamma
            early_stopping_metric='far',# 或 'far' / 'val_loss'
            patience=5,                # 早停耐心
            pos_class_id=1,            # 以 heymemo=1 为正类，计算 F1/FAR
            freeze_frontend=True       # 冻结 conv/rnn，只训练 fc
        )