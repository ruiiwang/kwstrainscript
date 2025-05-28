import os
import soundfile as sf
import numpy as np
import librosa

# 基准音频路径和参数
reference_audio = './mixed_audio.wav'
ref_data, ref_sr = sf.read(reference_audio)
ref_duration = len(ref_data) / ref_sr

# 更新后的分类识别部分
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

# 定义多个根目录
root_dirs = [
    "D:/project/AI-Glasses-Voice-Commands-Dataset/dataset",
]

# 修改主循环
for root_dir in root_dirs:
    if not os.path.exists(root_dir):
        print(f"警告: 目录 {root_dir} 不存在，跳过")
        continue
        
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.wav'):
                # 定义input_path
                input_path = os.path.join(root, file)
                
                # 分类逻辑保持不变
                category = 'UNKNOWN_WORD'
                for keyword in keywords_dict:
                    if keyword in file and keyword != 'UNKNOWN_WORD':
                        category = keyword
                        break
                
                # 直接输出到分类文件夹
                output_root = "D:/project/1.6svoice"  # 修改为你的输出根目录
                output_path = os.path.join(output_root, category, file)
                
                # 确保分类目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                try:
                    # 读取音频
                    data, sr = sf.read(input_path)
                    current_duration = len(data) / sr
                    
                    # 调整音频长度
                    # 修改音频处理部分
                    # 在音频处理部分添加采样率检查
                    if sr != 16000:
                        print(f"{input_path} 的采样率为{sr}Hz, 正在转换为16000Hz")
                        # 使用librosa进行高质量采样率转换
                        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                        sr = 16000
                    
                    # 然后继续原有的长度调整逻辑
                    if current_duration > ref_duration:
                        # 截断
                        data = data[:int(ref_duration * sr)]
                    elif current_duration < ref_duration:
                        # 在前面补静音
                        silence = np.zeros(int((ref_duration - current_duration) * sr))
                        data = np.concatenate([silence, data])  # 静音在前
                    
                    # 保存文件
                    sf.write(output_path, data, sr)
                    print(f"处理完成: {input_path} -> {output_path}")
                    
                except Exception as e:
                    print(f"处理失败 {input_path}: {str(e)}")
