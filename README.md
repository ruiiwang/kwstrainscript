# 语音关键词识别项目

## 项目概述
这是一个基于CRNN(卷积循环神经网络)的语音关键词识别系统，用于识别特定的语音命令。

## 主要功能
- 音频特征提取(MFCC)
- 训练CRNN模型
- 语音关键词分类

## 文件结构
├── converted_pickle/    # 转换后的特征数据（11类）
├── converted_pickle2/   # 转换后的特征数据（8类）
├── converted_pickle3/   # 转换后的误识别特征数据
├── model/               # 模型定义
│   ├── crnn_model.py    # CRNN模型实现
│   └── rnn_cells/       # 自定义RNN单元
├── origin_pickle/       # 原始特征数据（11类）
├── origin_pickle2/      # 原始特征数据 + 误识别特征数据（8类）
├── so_mfcc.py           # 音频特征提取代码
├── test_model.py        # 测试脚本
├── test_pkl.py          # 测试pkl数据
├── test_long.py         # 测试长音频（添加负样本）
├── train_crnn.py        # 训练脚本
├── checkpoint1/
│   ├── crnn_model_best.pth     # 训练好的模型（最佳）
│   └── training_log.txt        # 训练日志

## 快速开始
1. 训练模型
```
python train_crnn.py
```

2. 测试模型
```
python test_model.py    # 测试wav数据
python test_pkl.py      # 测试pkl数据
python test_long.py     # 测试长音频（添加负样本）
```

## 训练数据
- 8个关键词类别（已更新），1.6s 16000Hz 单通道wav音频
- 每个关键词对应的MFCC特征存储在pickle文件中

## 输出说明
- 训练过程中会输出每个epoch的损失和准确率
- 详细日志保存在 checkpoint{i}/training_log.txt
- 训练好的模型保存为 checkpoint{i}/crnn_model_best.pth
