import os
import librosa
import pickle
import torch
import numpy as np
from mfcc_io import mfcc

def process_folder(input_dir, pattern=None):
    features_list = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav') and (pattern is None or pattern in file):
                file_path = os.path.join(root, file)
                audio_data = librosa.load(file_path, sr=16000)[0]
                mfcc_feat = mfcc(
                    y=audio_data, sr=16000, n_mfcc=16, n_mels=40,
                    win_length=512, window='hamming', hop_length=256,
                    n_fft=512, fmin=20, fmax=4050
                )
                
                # 确保特征形状为(16,100)
                if mfcc_feat.shape[1] < 100:
                    mfcc_feat = np.pad(mfcc_feat, ((0,0),(0,100-mfcc_feat.shape[1])))
                elif mfcc_feat.shape[1] > 100:
                    mfcc_feat = mfcc_feat[:,:100]
                
                features_list.append(mfcc_feat)
    
    # 转换为三维张量(x,16,100)
    features_tensor = torch.FloatTensor(np.stack(features_list))
    return features_tensor

if __name__ == '__main__':
    input_folders = [
        # '/mnt/d/1.6svoice/HeyMemo',
        # '/mnt/d/1.6svoice/LookAnd',
        # '/mnt/d/1.6svoice/Next',
        # '/mnt/d/1.6svoice/Pause',
        # '/mnt/d/1.6svoice/Play',
        # '/mnt/d/1.6svoice/StopRecording',
        # '/mnt/d/1.6svoice/TakeAPicture',
        # '/mnt/d/1.6svoice/TakeAVideo',
        # '/mnt/d/1.6svoice/UNKNOWN_WORD',
        # '/mnt/d/1.6svoice/VolumeDown',
        # '/mnt/d/1.6svoice/VolumeUp',
        '/mnt/f/wrong_segments',
    ]
    
    # 添加文件名模式参数，例如只处理包含"20250611"的文件
    file_pattern = None
    # file_pattern = "2_2025"
    
    for folder in input_folders:
        folder_name = os.path.basename(folder)
        # output_file = f'{folder_name}_features.pkl'
        output_file = 'origin_pickle_un/UNKNOWN_WORD_features.pkl'
        
        print(f'正在处理文件夹: {folder}')
        features = process_folder(folder, pattern=file_pattern)
        
        with open(output_file, 'wb') as f:
            pickle.dump(features, f)
        print(f'已保存: {output_file}')
        print(f'特征张量形状: {features.shape}')  # 输出形状如torch.Size([4096, 16, 100])
        