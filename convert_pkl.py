import torch
import pickle
import os

# 关键词到标签的映射字典
keywords_dict = {
    'UNKNOWN_WORD': 0,
    'HeyMemo': 1,
    'Next': 2,
    'Pause': 3,
    'Play': 4,
    'StopRecording': 5,
    'TakeAPicture': 6,
    'TakeAVideo': 7,
    'VolumeDown': 8,
    'VolumeUp': 9,
    'LookAnd': 10
}

# 文件列表
file_list = [
    'HeyMemo_features.pkl',
    'Next_features.pkl',
    'LookAnd_features.pkl',
    'Pause_features.pkl',
    'Play_features.pkl',
    'StopRecording_features.pkl',
    'TakeAPicture_features.pkl',
    'TakeAVideo_features.pkl',
    'UNKNOWN_WORD_features.pkl',
    'VolumeDown_features.pkl',
    'VolumeUp_features.pkl'
]

# 从文件名提取关键词并生成标签
# 这里假设文件名格式为"keyword_xxx.pkl"，实际应根据您的文件名格式调整
def get_label_from_filename(filename):
    for keyword in keywords_dict:
        if keyword.lower() in filename.lower():
            return keywords_dict[keyword]
    return keywords_dict['UNKNOWN_WORD']

# 处理每个文件并单独保存
for file in file_list:
    with open(file, 'rb') as f:
        data = pickle.load(f)
    
    # 获取标签
    label = get_label_from_filename(file)
    
    # 假设每个文件的数据形状为(N,16,100)
    features = data
    labels = torch.full((features.shape[0],), label, dtype=torch.int32)
    
    # 创建输出数据
    output_data = (features, labels)
    
    # 生成输出文件名
    output_file = f"converted_{file}"
    
    # 保存转换后的数据
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
    
    # 打印处理信息
    print(f"Processed {file} -> {output_file}")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Label value: {label}")
    