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
    "fc_out": 11  # 11个类别
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
                feat = data[0] # 保持原始形状
                features.append(feat)
                # 将标签转换为 Long 类型
                labels.append(data[1].long())
    return torch.cat(features), torch.cat(labels)

def train_model(model, features, labels, epochs=10, batch_size=32):
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    log_message('开始训练...')
    log_message(f'总样本数: {len(features)}')
    log_message(f'批量大小: {batch_size}')
    log_message(f'训练周期: {epochs}')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        log_message(f'Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
    
    log_message('训练完成!')
    log_file.close()
    torch.save(model.state_dict(), 'crnn_model.pth')

if __name__ == "__main__":
    # 加载数据
    features, labels = load_all_pkls('converted_pickle')
    
    # 创建模型
    model = CnnRnnModel1Channel(config)
    
    # 开始训练
    train_model(model, features, labels, epochs=50, batch_size=64)
    