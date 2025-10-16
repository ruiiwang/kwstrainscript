import os
import librosa
import pickle
import torch
import numpy as np
from mfcc_io import mfcc

# 关键词到标签的映射
keywords_dict = {
    'UNKNOWN_WORD': 0,
    'HeyMemo': 1,
    # 'LookAnd': 2,
    # 'Pause': 3,
    # 'Play': 4,
    # 'StopRecording': 5,
    # 'TakeAPicture': 6,
    # 'TakeAVideo': 7,
}

def process_folder(input_dir, pattern=None):
    """
    处理文件夹中的.wav文件以提取MFCC特征。
    """
    features_list = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav') and (pattern is None or pattern in file):
                file_path = os.path.join(root, file)
                try:
                    audio_data = librosa.load(file_path, sr=16000)[0]
                    mfcc_feat = mfcc(
                        y=audio_data, sr=16000, n_mfcc=16, n_mels=40,
                        win_length=512, window='hamming', hop_length=256,
                        n_fft=512, fmin=20, fmax=4050
                    )
                    
                    # 确保特征形状为 (16, 100)
                    if mfcc_feat.shape[1] < 100:
                        mfcc_feat = np.pad(mfcc_feat, ((0, 0), (0, 100 - mfcc_feat.shape[1])))
                    elif mfcc_feat.shape[1] > 100:
                        mfcc_feat = mfcc_feat[:, :100]
                    
                    features_list.append(mfcc_feat)
                except Exception as e:
                    print(f"处理 {file_path} 时出错: {e}")
    
    if not features_list:
        return torch.FloatTensor([])
        
    # 转换为3D张量 (x, 16, 100)
    features_tensor = torch.FloatTensor(np.stack(features_list))
    return features_tensor

if __name__ == '__main__':
    # --- 配置 ---
    # 包含.wav文件的输入文件夹列表。
    # 文件夹的名称应与keywords_dict中的键相对应。
    input_folders = [
        # '/mnt/f/1.6s_voice/HeyMemo_augment',
        # '/mnt/f/1.6s_voice/LookAnd',
        # '/mnt/f/1.6s_voice/Next',
        # '/mnt/f/1.6s_voice/Pause',
        # '/mnt/f/1.6s_voice/Play',
        # '/mnt/f/1.6s_voice/StopRecording',
        # '/mnt/f/1.6s_voice/TakeAPicture',
        # '/mnt/f/1.6s_voice/TakeAVideo',
        # '/mnt/f/1.6s_voice/UNKNOWN_WORD',
        # '/mnt/f/1.6s_voice/VolumeDown',
        # '/mnt/f/1.6s_voice/VolumeUp',
        '/mnt/f/1.6s_voice/CHiME6/S07',
        '/mnt/f/1.6s_voice/CHiME6/S08',
        '/mnt/f/1.6s_voice/CHiME6/S12',
        '/mnt/f/1.6s_voice/CHiME6/S13',
    ]
    
    # 用于保存最终.pkl文件的目录
    output_dir = 'converted_2_ft2'
    
    # 可选：仅处理包含此模式的文件
    file_pattern = None
    # file_pattern = "2_2025"
    # --- 配置结束 ---

    os.makedirs(output_dir, exist_ok=True)
    
    for folder_path in input_folders:
        folder_name = os.path.basename(folder_path)
        print(f"正在处理文件夹: {folder_path}")
        
        # 步骤1：从文件夹中的.wav文件中提取特征
        features = process_folder(folder_path, pattern=file_pattern)
        
        if features.shape[0] == 0:
            print(f"  在 {folder_path} 中找不到匹配的.wav文件。正在跳过。")
            continue
            
        # 步骤2：根据文件夹名称确定标签
        # 如果文件夹名称不在字典中，则默认为UNKNOWN_WORD
        label = keywords_dict.get(folder_name, keywords_dict['UNKNOWN_WORD'])
        labels = torch.full((features.shape[0],), label, dtype=torch.int32)
        
        # 步骤3：合并特征和标签并保存到.pkl文件
        output_data = (features, labels)
        
        output_filename = f"{folder_name}_data.pkl"
        output_filepath = os.path.join(output_dir, output_filename)
        
        with open(output_filepath, 'wb') as f:
            pickle.dump(output_data, f)
            
        print(f"  成功保存到 {output_filepath}")
        print(f"    特征形状: {features.shape}")
        print(f"    标签形状:   {labels.shape}")
        print(f"    标签值:    {label} (关键词 '{folder_name}')")