import os
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model.crnn_model import CnnRnnModel1Channel

# 模型配置
# config = {
#     "in_c": 16,
#     "conv": [{"out_c": 32, "k": 16, "s": 2, "p":5, "dropout": 0.0},
#              {"out_c": 64, "k": 8, "s": 2, "p":3, "dropout": 0.0}],
#     "rnn": {"dim": 64, "layers": 1, "dropout": 0.25, "bidirectional": True},
#     "fc_out": 8  # 8个类别
# }
config = {
    "in_c": 16,
    "conv": [{"out_c": 16, "k": 8, "s": 2, "p": 1, "dropout": 0.0},
            {"out_c": 32, "k": 4, "s": 2, "p": 1, "dropout": 0.0}],
    "rnn": {"dim": 32, "layers": 1, "dropout": 0.2, "bidirectional": True},
    "fc_out": 8
}

def log_message(message, log_file):
    print(message)
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

def train_model(model, features, labels, epochs=10, batch_size=32, folder='checkpoint', resume_checkpoint=None):
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
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 确保checkpoint目录存在
    if not os.path.exists(folder):
        os.makedirs(folder)
    # 初始化日志文件
    log_file = open(os.path.join(folder, 'training_log.txt'), 'a')
    start_epoch = 0
    best_loss = float('inf')

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        log_message(f'从检查点 {resume_checkpoint} 恢复训练...', log_file)
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        log_message(f'从 Epoch {start_epoch + 1} 开始训练，上次最佳损失: {best_loss:.4f}', log_file)
    else:
        log_message('开始训练...', log_file)
        log_message(f'总样本数: {len(features)}', log_file)
        log_message(f'batch_size: {batch_size}', log_file)
        log_message(f'训练周期: {epochs}', log_file)

    # 确保checkpoint目录存在
    if not os.path.exists(folder):
        os.makedirs(folder)

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
        epoch_acc = 100. * correct / total
        log_message(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%', log_file)
        
        # 保存checkpoint
        checkpoint_dict = {
            "epoch": epoch+1,
            "loss": epoch_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        torch.save(checkpoint_dict, f'{folder}/epoch{epoch+1}.pth')
        
        # 更新最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model.state_dict()
            torch.save(best_model, f'{folder}/crnn_model_best.pth')
    
    log_message(f'Best model saved with loss: {best_loss:.4f}', log_file)
    log_message('训练完成!', log_file)
    log_file.close()
    
    # 删除原来的保存最终模型的代码
    # torch.save(model.state_dict(), 'crnn_model.pth')  # 这行已被删除

if __name__ == "__main__":
    # 设置要加载的pkl文件目录列表
    # 您可以将新的pkl文件目录添加到这个列表中
    pkl_data_dirs = ['converted_pickle2'] # 添加您的新数据目录
    
    # 加载数据
    features, labels = load_all_pkls(pkl_data_dirs)
    
    # 创建模型
    model = CnnRnnModel1Channel(config)
    
    # 设置要恢复的检查点路径，如果没有则设置为 None
    resume_checkpoint_path = None # 例如: 'checkpoint1/epoch15.pth'

    # 开始训练
    train_model(model, features, labels, epochs=100, batch_size=64, folder='checkpoint3', resume_checkpoint=resume_checkpoint_path)
    