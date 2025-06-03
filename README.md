# 语音关键词识别项目

## 项目概述
这是一个基于CRNN(卷积循环神经网络)的语音关键词识别系统，用于识别特定的语音命令。

## 主要功能
- 音频特征提取(MFCC)
- 训练CRNN模型
- 语音关键词分类

## 文件结构
├── converted_pickle/    # 转换后的特征数据
├── model/               # 模型定义
│   ├── crnn_model.py    # CRNN模型实现
│   └── rnn_cells/       # 自定义RNN单元
├── origin_pickle/       # 原始特征数据
├── train_crnn.py       # 训练脚本
├── crnn_model.pth      # 训练好的模型
└── training_log.txt    # 训练日志

## 快速开始
1. 训练模型
```
python train_crnn.py
```
2. 使用训练好的模型
```
from model.crnn_model import CnnRnnModel1Channel
import torch

# 加载模型
model = CnnRnnModel1Channel(config)
model.load_state_dict(torch.load('crnn_model.pth'))
model.eval()

# 输入数据应为形状(batch_size, 100, 16)的tensor
with torch.no_grad():
    output = model(input_data)
    predicted_class = torch.argmax(output).item()
```

## 训练数据
- 11个关键词类别
- 每个关键词对应的MFCC特征存储在pickle文件中

## 输出说明
- 训练过程中会输出每个epoch的损失和准确率
- 详细日志保存在 training_log.txt
- 训练好的模型保存为 crnn_model.pth
