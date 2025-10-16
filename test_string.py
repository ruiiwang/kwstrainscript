import os
import torch
import librosa
import soundfile as sf
from collections import deque, Counter
from model.crnn_model import CnnRnnModel1Channel
from mfcc_io import mfcc
import numpy as np
from wake_strategies import StrategyFactory, ConsecutiveStrategy, AverageStrategy, PeakStrategy, ComboStrategy

# 假设你的mfcc_io和model.crnn_model模块已正确配置

# --- 配置部分 ---
# 模型配置 (与test_model.py一致)
class_names = {
    0: "UNKNOWN_WORD",
    1: "HeyMemo",
    # 2: "LookAnd",
    # 3: "Pause",
    # 4: "Play",
    # 5: "StopRecording",
    # 6: "TakeAPicture",
    # 7: "TakeAVideo"
}

config = {
    "in_c": 16,
    "conv": [{"out_c": 16, "k": 8, "s": 2, "p": 1, "dropout": 0.0},
            {"out_c": 32, "k": 4, "s": 2, "p": 1, "dropout": 0.0}],
    "rnn": {"dim": 32, "layers": 1, "dropout": 0.2, "bidirectional": True},
    "fc_out": len(class_names)
}

# 流式仿真参数
SR = 16000
WINDOW_LENGTH = 1.6  # seconds
HOP_LENGTH = 0.02   # seconds, 20ms
AUDIO_BUFFER_SIZE = int(WINDOW_LENGTH * SR)
HOP_SAMPLES = int(HOP_LENGTH * SR)

# --- 模型加载 ---
def load_model(model_path):
    """Loads the pre-trained PyTorch model."""
    model = CnnRnnModel1Channel(config)
    try:
        model.load_state_dict(torch.load(model_path))
    except KeyError:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# --- Core Streaming Simulation Class ---
class StreamProcessor:
    def __init__(self, model):
        self.model = model
        # Audio buffer for the sliding window, initialized with float zeros
        self.audio_buffer = deque([0.0] * AUDIO_BUFFER_SIZE, maxlen=AUDIO_BUFFER_SIZE)
        self.total_processed_time = 0.0
        self.triggered = False  # 标记当前文件是否已触发唤醒
        self.trigger_strategy = None  # 记录触发的策略类型
        self.trigger_word = None  # 记录触发的唤醒词
        
        # 每个词的唤醒策略配置
        self.word_strategies = {
            "HeyMemo": {
                "strategies": [
                    ConsecutiveStrategy(consecutive_threshold=0.7, consecutive_window_ms=1800),
                    AverageStrategy(average_threshold=0.75, average_window_ms=3200),
                    PeakStrategy(peak_threshold=0.75, duration_threshold=0.7, duration_window_ms=2400),
                    ComboStrategy(peak_threshold=0.75, avg_threshold=0.7, duration_ms=2000, weights=None)
                ]
            },
            # 可以为其他词添加策略
        }
        
        # 最小唤醒间隔（秒）
        self.min_interval = 1.6

    def process_chunk(self, chunk):
        """Processes a chunk of audio and returns model probabilities."""
        # Add new data to the buffer
        self.audio_buffer.extend(chunk)
        self.total_processed_time += HOP_LENGTH

        # Get the current window of audio data, ensuring float32 type
        current_window = np.array(self.audio_buffer, dtype=np.float32)

        # Feature extraction
        mfcc_data = mfcc(y=current_window, sr=SR, n_mfcc=16, n_mels=40, 
                        win_length=512, window='hamming', hop_length=HOP_SAMPLES,
                        fmin=20, fmax=4050)
        
        # Add batch dimension for the model
        features = torch.FloatTensor(mfcc_data).unsqueeze(0)
        
        # Model prediction
        with torch.no_grad():
            outputs = self.model(features)
            probs = torch.softmax(outputs, 1).squeeze(0)
            
        return probs, current_window

    def update_state(self, probs, current_window, output_dir, log_file, wav_path=None):
        """检测所有词的概率并根据策略触发唤醒"""
        # 检查每个词是否满足唤醒条件
        for i in range(1, len(class_names)):  # 跳过UNKNOWN_WORD
            word = class_names[i]
            prob = probs[i].item()
            
            if word in self.word_strategies:
                # 更新每个策略的历史记录
                for strategy in self.word_strategies[word]["strategies"]:
                    strategy.update_history(prob, self.total_processed_time)
                
                # 检查每个策略是否触发
                for strategy in self.word_strategies[word]["strategies"]:
                    triggered, trigger_type = strategy.check_trigger(self.total_processed_time)
                    
                    if triggered and not self.triggered:
                        # 标记当前文件已触发唤醒
                        self.triggered = True
                        self.trigger_strategy = f"{strategy.name}-{trigger_type}"
                        self.trigger_word = word
                        
                        # 记录到日志文件
                        with open(log_file, 'a') as f:
                            f.write(f"{wav_path} {self.total_processed_time:.2f}s ({word}) ({strategy.name}-{trigger_type})\n")
                        
                        # 一旦有策略触发，就不再检查其他策略
                        break
    
    def reset(self):
        """重置所有状态"""
        # 重置所有策略
        for word in self.word_strategies:
            for strategy in self.word_strategies[word]["strategies"]:
                strategy.reset()
        
        # 填充缓冲区以重新开始
        self.audio_buffer = deque([0.0] * AUDIO_BUFFER_SIZE, maxlen=AUDIO_BUFFER_SIZE)
        self.triggered = False  # 重置触发状态
        self.trigger_strategy = None
        self.trigger_word = None

# --- Simulation entry point ---
def run_simulation(model, wav_path, output_dir, log_file):
    """Simulates streaming processing from a file or directory."""
    # 检查路径是文件还是目录
    if os.path.isdir(wav_path):
        # 如果是目录，遍历所有WAV文件
        print(f"处理目录: {wav_path}")
        processed_files = 0
        triggered_files = 0
        strategy_counter = Counter()  # 用于统计各类策略的唤醒次数
        
        for root, dirs, files in os.walk(wav_path):
            wav_files = [f for f in files if f.endswith('.wav')]
            total_files = len(wav_files)
            
            for file in wav_files:
                file_path = os.path.join(root, file)
                triggered, strategy_info = process_audio_file(model, file_path, output_dir, log_file)
                processed_files += 1
                if triggered:
                    triggered_files += 1
                    if strategy_info:
                        strategy_counter[strategy_info] += 1
        
        # 计算唤醒率
        wake_rate = (triggered_files / processed_files * 100) if processed_files > 0 else 0
        print(f"\n唤醒统计结果:")
        print(f"成功唤醒文件数: {triggered_files}/{processed_files}")
        print(f"唤醒率: {wake_rate:.2f}%")
        
        # 输出各类策略的唤醒次数
        print(f"\n各类唤醒策略统计:")
        for strategy, count in strategy_counter.items():
            strategy_rate = (count / triggered_files * 100) if triggered_files > 0 else 0
            print(f"{strategy}: {count}次 ({strategy_rate:.2f}%)")
        
        # 记录到日志文件
        with open(log_file, 'a') as f:
            f.write(f"\n唤醒统计结果: {wav_path}\n")
            f.write(f"成功唤醒文件数: {triggered_files}/{processed_files}\n")
            f.write(f"唤醒率: {wake_rate:.2f}%\n")
            f.write(f"\n各类唤醒策略统计:\n")
            for strategy, count in strategy_counter.items():
                strategy_rate = (count / triggered_files * 100) if triggered_files > 0 else 0
                f.write(f"{strategy}: {count}次 ({strategy_rate:.2f}%)\n")
            f.write("\n")
            
    else:
        # 如果是单个文件，直接处理
        triggered, strategy_info = process_audio_file(model, wav_path, output_dir, log_file)
        print(f"\n唤醒结果: {'成功' if triggered else '失败'}")
        if triggered and strategy_info:
            print(f"触发策略: {strategy_info}")

def process_audio_file(model, wav_path, output_dir, log_file):
    """处理单个音频文件，返回是否触发唤醒和触发的策略"""
    if not os.path.exists(wav_path):
        print(f"Error: File not found at {wav_path}")
        return False, None

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    stream_processor = StreamProcessor(model)
    try:
        audio, _ = librosa.load(wav_path, sr=SR)
        
        # Simulate streaming input by iterating over the audio in chunks
        for i in range(0, len(audio), HOP_SAMPLES):
            chunk = audio[i:i + HOP_SAMPLES]
            
            # Pad the last chunk if it's incomplete
            if len(chunk) < HOP_SAMPLES:
                chunk = np.pad(chunk, (0, HOP_SAMPLES - len(chunk)), 'constant')
                
            probs, current_window = stream_processor.process_chunk(chunk)
            if probs is not None:
                # 传递文件路径给update_state方法
                stream_processor.update_state(probs, current_window, output_dir, log_file, wav_path)
        
        return stream_processor.triggered, stream_processor.trigger_strategy  # 返回是否触发唤醒和触发的策略
    
    except Exception as e:
        print(f"处理文件 {wav_path} 时出错: {str(e)}")
        return False, None

if __name__ == "__main__":
    # Load the model
    # Note: Replace with the actual path to your model file
    model_path = 'checkpoint_2.2_ft/crnn_model_best.pth'
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please check the path.")
        exit()

    model = load_model(model_path)
    
    # Set up simulation paths
    input_audio_path = '/mnt/f/string_test/heymemo/'  # 目录路径
    output_folder = '/mnt/f/string_test/heymemo_output/'
    log_file = './prediction_2.log'           # 日志文件
    
    # --- Example Usage ---
    # To run a positive test (with wakeup words)
    # print(f"--- 开始流式仿真: {input_audio_path} ---")
    # run_simulation(model, input_audio_path, output_folder, log_file)

    # To run a negative test (for false alarms)
    noise_audio_path = '/mnt/f/wrong_segments2/'
    print(f"\n--- 开始误唤醒仿真: {noise_audio_path} ---")
    run_simulation(model, noise_audio_path, output_folder, log_file)
