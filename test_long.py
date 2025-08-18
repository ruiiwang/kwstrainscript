import os
import torch
import librosa
import soundfile as sf
from datetime import datetime
from model.crnn_model import CnnRnnModel1Channel
from mfcc_io import mfcc

# 模型配置(与test_model.py一致)
config = {
    "in_c": 16,
    "conv": [{"out_c": 16, "k": 8, "s": 2, "p": 1, "dropout": 0.0},
                {"out_c": 32, "k": 4, "s": 2, "p": 1, "dropout": 0.0}],
    "rnn": {"dim": 32, "layers": 1, "dropout": 0.2, "bidirectional": True},
    "fc_out": 8
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
    # 修改此处，加载整个检查点字典，然后提取model_state_dict
    try:
        model.load_state_dict(torch.load(model_path))
    except KeyError:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def process_audio(model, wav_path, output_dir, log_file, last_wakeup_audio_time):
    # 参数设置
    sr = 16000
    window_length = 1.6  # 秒
    hop_length = 0.1    # 秒
    threshold = 0.7
    min_interval = 1.6   # 最小间隔时间，单位秒
    
    # 加载音频
    audio, _ = librosa.load(wav_path, sr=sr)
    # 添加音量归一化
    audio = librosa.util.normalize(audio, norm=2)
    
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
            probs = torch.softmax(outputs, 1)  # 将输出转换为概率
            max_prob, predicted = torch.max(probs, 1)
            # 添加阈值判断
            predicted_class = predicted.item() if max_prob > threshold else 0  # 0对应UNKNOWN_WORD
            
        # 如果不是UNKNOWN_WORD则保存片段
        if predicted_class != 0:
            current_audio_time = i / sr # 当前唤醒在音频中的时间点
            # 检查与上次唤醒的时间间隔
            if (current_audio_time - last_wakeup_audio_time) < min_interval:
                continue # 如果间隔太短，则跳过本次唤醒

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = os.path.join(output_dir, 
                                     f"{predicted_class}_{timestamp}.wav")
            sf.write(output_path, segment, sr)
            
            # 记录日志
            with open(log_file, 'a') as f:
                f.write(f"{wav_path} -> {output_path} ({class_names[predicted_class]})\n")
            last_wakeup_audio_time = current_audio_time # 更新上次唤醒时间
    return last_wakeup_audio_time

def process_folder(model, input_dir, output_dir, log_file):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化上次唤醒时间为0，确保第一次唤醒能被记录
    last_wakeup_audio_time = 0.0

    # 遍历所有wav文件
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_path = os.path.join(root, file)
                # 对于每个新的音频文件，重置 last_wakeup_audio_time
                last_wakeup_audio_time = 0.0 
                last_wakeup_audio_time = process_audio(model, wav_path, output_dir, log_file, last_wakeup_audio_time)

if __name__ == "__main__":
    # 加载模型
    model = load_model('checkpoint2/crnn_model_best.pth')
    
    # 设置路径
    input_folder = '/mnt/f/voxceleb_data/voxceleb1/'  # 输入文件夹
    # input_folder = '/mnt/f/voiceprint_2/'
    # 已经提取完成了id10122的部分
    # 第二部分：id10123-id10471
    output_folder = '/mnt/f/wrong_segments2/'     # 输出文件夹
    log_file = './prediction2.log'           # 日志文件
    
    # 处理文件夹
    # process_folder(model, input_folder, output_folder, log_file)
    for i in range(1, 12000):
        input_folders = os.path.join(input_folder, f'id{i}')
        print(f"Processing folder: {input_folders}")
        # 处理文件夹
        process_folder(model, input_folders, output_folder, log_file)
