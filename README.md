# 语音关键词识别项目

## 项目概述
这是一个基于CRNN(卷积循环神经网络)的语音关键词识别系统，用于识别特定的语音命令。

## 主要功能
- 音频特征提取(MFCC)
- 训练CRNN模型
- 语音关键词分类

## 文件结构
```text
├── checkpoint/_old/        # 模型训练结果（config_20250609）
│   ├── checkpoint_8*/      # 8分类模型（config_20250815）
│   ├── checkpoint_2*/      # 2分类模型（config_20250909）
│   ├── checkpoint_1*/      # 1分类模型（config_20251017）
│   ├── checkpoint_1.1/     # 1分类-2输出模型（config_20250909）
│   └── checkpoint_2.2_ft2/ # 2分类模型（config_20250909）（现版本）
├── config/                 # 模型配置文件
├── converted_11/           # 转换后的特征数据（11类）
├── converted_8/            # 转换后的特征数据（8类）
├── converted_2*/           # 2分类模型训练的特征数据
├── converted_un/           # 转换后的误识别特征数据
├── model/                  # 模型定义
│   ├── crnn_model.py       # CRNN模型实现
│   └── rnn_cells/          # 自定义RNN单元
├── origin_pickle/          # 原始特征数据
├── quantproject*/          # 量化工程
│   ├── quant_public.py     # 量化代码
│   ├── test.py             # 测试量化模型
│   └── analyze.py          # 分析量化模型测试结果
├── batch_extract_mfcc.py   # 批量提取MFCC特征（已弃用）
├── convert_pkl.py          # 转化MFCC特征为pkl文件（已弃用）
├── batch_extract_pkl.py    # 批量提取MFCC特征并保存为可直接使用的pkl文件
├── mix_pkl.py              # 生成正负样本混合的pkl文件
├── so_mfcc.py              # 音频特征提取代码
├── pic_string.py           # 长音频流式仿真+画图（带策略唤醒）
├── pic_string_fq.py        # 长音频流式仿真+画图（float+quant）
├── wake_strategies.py      # 唤醒策略
├── test_string.py          # 长音频流式仿真
├── test_model.py           # 测试脚本
├── test_pkl.py             # 测试pkl数据
├── test_long.py            # 测试长音频（添加负样本）
├── test_one_class.py       # 1分类测试脚本
├── train_crnn.py           # 训练脚本
├── train_one_class.py      # 1分类训练脚本
├── train_dual_output.py    # 1分类-2输出训练脚本

```

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
- 1.6s 16000Hz 单通道wav音频
- 每个关键词对应的MFCC特征存储在pickle文件中

## 输出说明
- 训练过程中会输出每个epoch的损失和准确率
- 详细日志保存在 checkpoint{i}/training_log.txt
- 训练好的模型保存为 checkpoint{i}/crnn_model_best.pth
