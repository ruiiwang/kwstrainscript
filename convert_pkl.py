import torch
import pickle
import os

# 关键词到标签的映射字典
keywords_dict = {
    'UNKNOWN_WORD': 0,
    'HeyMemo': 1,
    'LookAnd': 2,
    'Pause': 3,
    'Play': 4,
    'StopRecording': 5,
    'TakeAPicture': 6,
    'TakeAVideo': 7,
}

# 原始文件所在目录
input_dir = 'origin_pickle_un'  # 修改为你的原始文件目录

# 获取目录下所有pkl文件
file_list = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]

# 从文件名提取关键词并生成标签
def get_label_from_filename(filename):
    for keyword in keywords_dict:
        if keyword.lower() in filename.lower():
            return keywords_dict[keyword]
    return keywords_dict['UNKNOWN_WORD']

# 处理每个文件并单独保存
for file in file_list:
    with open(os.path.join(input_dir, file), 'rb') as f:  # 使用完整路径
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
    
    # 确保输出目录存在
    os.makedirs('converted_un', exist_ok=True)
    
    # 保存转换后的数据
    with open(os.path.join('converted_un', output_file), 'wb') as f:
        pickle.dump(output_data, f)
    
    # 打印处理信息
    print(f"Processed {file} -> {output_file}")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Label value: {label}")
    