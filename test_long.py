import os
import torch
import librosa
import soundfile as sf
from datetime import datetime
from model.crnn_model import CnnRnnModel1Channel
from so_mfcc import mfcc

# 模型配置(与test_model.py一致)
config = {
    "in_c": 16,
    "conv": [
        {"out_c": 32, "k": 16, "s": 2, "p":5, "dropout": 0.0},
        {"out_c": 64, "k": 8, "s": 2, "p":3, "dropout": 0.0}
    ],
    "rnn": {"dim": 64, "layers": 1, "dropout": 0.25, "bidirectional": True},
    "fc_out": 8  # 8个类别
}

class_names = {
    0: "UNKNOWN_WORD",
    1: "HeyMemo",
    2: "LookAnd",
    3: "Pause",
    4: "Play",
    5: "StopRecording",
    6: "TakeAPicture",
    7: "TakeAVideo"
}

def load_model(model_path):
    model = CnnRnnModel1Channel(config)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def process_audio(model, wav_path, output_dir, log_file):
    # 参数设置
    sr = 16000
    window_length = 1.6  # 秒
    hop_length = 0.02    # 秒
    threshold = 0.7
    
    # 加载音频
    audio, _ = librosa.load(wav_path, sr=sr)
    
    # 计算采样点数
    window_samples = int(window_length * sr)
    hop_samples = int(hop_length * sr)
    
    # 滑动窗口处理
    for i in range(0, len(audio) - window_samples + 1, hop_samples):
        segment = audio[i:i+window_samples]
        
        # 提取特征
        mfcc_data = mfcc(y=segment, sr=sr, n_mfcc=16, n_mels=40, 
                        win_length=512, window='hamming', hop_length=256,
                        fmin=20, fmax=4050)  # 添加fmin和fmax参数
        features = torch.FloatTensor(mfcc_data).unsqueeze(0)
        
        # 预测
        with torch.no_grad():
            outputs = model(features)
            # _, predicted = torch.max(outputs, 1)
            # predicted_class = predicted.item()

            probs = torch.softmax(outputs, 1)  # 将输出转换为概率
            max_prob, predicted = torch.max(probs, 1)
            # 添加阈值判断
            predicted_class = predicted.item() if max_prob > threshold else 0  # 0对应UNKNOWN_WORD
            
        # 如果不是UNKNOWN_WORD则保存片段
        if predicted_class != 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = os.path.join(output_dir, 
                                     f"{predicted_class}_{timestamp}.wav")
            sf.write(output_path, segment, sr)
            
            # 记录日志
            with open(log_file, 'a') as f:
                f.write(f"{wav_path} -> {output_path} ({class_names[predicted_class]})\n")

def process_folder(model, input_dir, output_dir, log_file):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历所有wav文件
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_path = os.path.join(root, file)
                process_audio(model, wav_path, output_dir, log_file)

if __name__ == "__main__":
    # 加载模型
    model = load_model('8class_model_best.pth')
    
    # 设置路径
    input_folder = '/mnt/d/project/voxceleb_trainer/data/voxceleb1/'  # 输入文件夹
    # 已经提取完成了id10122的部分
    output_folder = './wrong_segments/'     # 输出文件夹
    log_file = './prediction.log'           # 日志文件
    
    # 处理文件夹
    process_folder(model, input_folder, output_folder, log_file)
