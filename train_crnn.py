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
    "conv": [
        {"out_c": 32, "k": 16, "s": 2, "p":5, "dropout": 0.0},
        {"out_c": 64, "k": 8, "s": 2, "p":3, "dropout": 0.0}
    ],
    "rnn": {"dim": 64, "layers": 1, "dropout": 0.25, "bidirectional": True},
    "fc_out": 8  # 8个类别
}

# 初始化日志文件
log_file = open('training_log.txt', 'w')

def log_message(message):
    print(message)
    log_file.write(message + '\n')
    log_file.flush()

# 加载所有pkl文件
def load_all_pkls(pkl_dir):
    features = []
    labels = []
    for pkl_file in os.listdir(pkl_dir):
        if pkl_file.endswith('.pkl'):
            with open(os.path.join(pkl_dir, pkl_file), 'rb') as f:
                data = pickle.load(f)
                feat = data[0]
                features.append(feat)
                labels.append(data[1].long())
    return torch.cat(features), torch.cat(labels)

def train_model(model, features, labels, epochs=10, batch_size=32):
    # 划分数据集: 60%训练集, 20%验证集, 20%测试集
    dataset_size = len(features)
    train_size = int(0.6 * dataset_size)
    val_size = int(0.2 * dataset_size)
    # test_size = dataset_size - train_size - val_size
    
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
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    log_message('开始训练...')
    log_message(f'总样本数: {len(features)}')
    log_message(f'批量大小: {batch_size}')
    log_message(f'训练周期: {epochs}')
    
    best_loss = float('inf')
    best_model = None
    
    # 确保checkpoint目录存在
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    for epoch in range(epochs):
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
        epoch_acc = 100. * correct / total
        log_message(f'Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
        
        # 保存checkpoint
        checkpoint_dict = {
            "epoch": epoch+1,
            "loss": epoch_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        torch.save(checkpoint_dict, f'checkpoint/epoch{epoch+1}.pth')
        
        # 更新最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model.state_dict()
            torch.save(best_model, 'checkpoint/crnn_model_best.pth')
            log_message(f'New best model saved with loss: {best_loss:.4f}')
    
    log_message('训练完成!')
    log_file.close()
    
    # 删除原来的保存最终模型的代码
    # torch.save(model.state_dict(), 'crnn_model.pth')  # 这行已被删除

if __name__ == "__main__":
    # 加载数据
    features, labels = load_all_pkls('converted_pickle2')
    
    # 创建模型
    model = CnnRnnModel1Channel(config)
    
    # 开始训练
    train_model(model, features, labels, epochs=30, batch_size=64)
    