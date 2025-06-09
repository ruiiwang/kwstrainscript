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
    "fc_out":  8 # 8个类别
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
    test_size = dataset_size - train_size - val_size
    
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
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    log_message(f'总样本数: {dataset_size}')
    log_message(f'训练集: {train_size}, 验证集: {val_size}, 测试集: {test_size}')
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 训练阶段
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        log_message(f'Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'crnn_model_best.pth')
    
    # 测试阶段
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
    
    test_acc = 100. * test_correct / test_total
    log_message(f'测试集准确率: {test_acc:.2f}%')
    log_message('训练完成!')
    log_file.close()
    torch.save(model.state_dict(), 'crnn_model.pth')

if __name__ == "__main__":
    # 加载数据
    features, labels = load_all_pkls('converted_pickle')
    
    # 创建模型
    model = CnnRnnModel1Channel(config)
    
    # 开始训练
    train_model(model, features, labels, epochs=50, batch_size=1024)
    