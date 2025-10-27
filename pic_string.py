import os
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, Counter
from model.crnn_model import CnnRnnModel1Channel
from mfcc_io import mfcc
from wake_strategies import ConsecutiveStrategy, AverageStrategy, PeakStrategy, ComboStrategy

# 统一模型配置和类别名称，便于管理
class Config:
    SR = 16000
    WINDOW_LENGTH_S = 1.6
    HOP_LENGTH_S = 0.02
    AUDIO_BUFFER_SIZE = int(WINDOW_LENGTH_S * SR)
    HOP_SAMPLES = int(HOP_LENGTH_S * SR)
    
    # 专门用于MFCC特征提取的hop_length和win_length，与模型训练时保持一致
    MFCC_HOP_LENGTH = 256
    MFCC_WIN_LENGTH = 512

    # 模型配置
    CLASS_NAMES = {
        0: "UNKNOWN_WORD",
        1: "HeyMemo",
    }
    CLASS_NAMES_INDEX = {name: index for index, name in CLASS_NAMES.items()}
    
    MODEL_CONFIG = {
        "in_c": 16,
        "conv": [{"out_c": 16, "k": 8, "s": 2, "p": 1, "dropout": 0.0},
                 {"out_c": 32, "k": 4, "s": 2, "p": 1, "dropout": 0.0}],
        "rnn": {"dim": 32, "layers": 1, "dropout": 0.2, "bidirectional": True},
        "fc_out": len(CLASS_NAMES),
    }
    

class StreamSimulator:
    """
    流式仿真类，管理音频流处理和模型推理（已优化为不等步长增量更新策略）。
    """
    def __init__(self, model_path, config):
        self.config = config
        self.model = self._load_model(model_path)
        self.model_input_mfcc_length = self._calculate_mfcc_length()
        
        # 每个词的唤醒策略配置
        self.word_strategies = {
            "HeyMemo": {
                "strategies": [
                    PeakStrategy(peak_threshold=0.8, duration_threshold=0.7, duration_window_ms=320),
                    ComboStrategy(peak_threshold=0.8, avg_threshold=0.7, duration_ms=320, weights=None),
                    AverageStrategy(average_threshold=0.9, average_window_ms=400),
                    ConsecutiveStrategy(consecutive_threshold=0.75, consecutive_window_ms=320)
                ]
            }
        }
        self.triggered_events = []

    def _load_model(self, model_path):
        """加载模型并返回"""
        model = CnnRnnModel1Channel(self.config.MODEL_CONFIG)
        try:
            model.load_state_dict(torch.load(model_path))
        except KeyError:
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
        
    def _get_mfcc_from_audio(self, audio_data):
        """从给定的音频数据中提取MFCC特征的辅助函数"""
        return mfcc(y=audio_data, sr=self.config.SR, n_mfcc=16, n_mels=40, 
                    win_length=512, window='hamming', 
                    hop_length=self.config.MFCC_HOP_LENGTH, n_fft=512,
                    fmin=20, fmax=4050)

    def _calculate_mfcc_length(self):
        """计算模型需要的MFCC特征长度"""
        zero_audio = np.zeros(self.config.AUDIO_BUFFER_SIZE, dtype=np.float32)
        mfccs = self._get_mfcc_from_audio(zero_audio)
        return mfccs.shape[1]

    def process_audio_stream(self, wav_path):
        """
        处理音频流并进行模型推理（不等步长增量更新版本）。
        """
        audio, sr = librosa.load(wav_path, sr=self.config.SR)
        
        for strategy in self.word_strategies["HeyMemo"]["strategies"]:
            strategy.reset()
        self.triggered_events = []

        probabilities, timestamps, rms_values = [], [], []

        # --- 初始化状态 (已修正) ---
        # 1. 1.6s音频缓冲区，初始为全0，模拟寂静状态
        audio_buffer = deque([0.0] * self.config.AUDIO_BUFFER_SIZE, maxlen=self.config.AUDIO_BUFFER_SIZE)
        buffer_array = np.array(audio_buffer, dtype=np.float32)
        
        # 2. 首次为全0的音频计算一次完整的MFCC矩阵
        mfcc_matrix = self._get_mfcc_from_audio(buffer_array)
        
        # 3. 初始化用于不等步长更新的临时缓冲区
        unprocessed_audio = []

        # --- 开始流式处理 ---
        # 以20ms (320 samples)为步长
        for i in range(0, len(audio), self.config.HOP_SAMPLES):
            # 获取当前20ms块
            new_samples_chunk = audio[i : i + self.config.HOP_SAMPLES]
            
            if not new_samples_chunk.any():
                continue
            
            # 将新的20ms音频块加入临时缓冲区
            unprocessed_audio.extend(new_samples_chunk)

            # 计算可以生成多少个新的16ms MFCC帧
            num_frames_to_update = len(unprocessed_audio) // self.config.MFCC_HOP_LENGTH
            
            if num_frames_to_update > 0:
                # 循环更新MFCC矩阵
                for _ in range(num_frames_to_update):
                    # 取出下一个16ms的音频帧用于更新
                    frame_to_process = unprocessed_audio[:self.config.MFCC_HOP_LENGTH]
                    
                    # 更新主音频缓冲区 (deque会自动处理滑动)
                    audio_buffer.extend(frame_to_process)
                    
                    # --- 增量更新MFCC ---
                    buffer_array = np.array(audio_buffer, dtype=np.float32)
                    new_frame_audio = buffer_array[-512:] # win_length is 512
                    new_mfcc_frames = self._get_mfcc_from_audio(new_frame_audio)
                    new_mfcc_column = new_mfcc_frames[:, -1]
                    
                    mfcc_matrix = np.roll(mfcc_matrix, -1, axis=1)
                    mfcc_matrix[:, -1] = new_mfcc_column

                # 更新临时缓冲区，移除已处理的音频
                processed_samples = num_frames_to_update * self.config.MFCC_HOP_LENGTH
                unprocessed_audio = unprocessed_audio[processed_samples:]

            # --- 在每个20ms步长后，都进行一次推理 ---
            if mfcc_matrix.shape[1] != self.model_input_mfcc_length:
                print(f"MFCC长度不匹配: {mfcc_matrix.shape[1]}, 期望: {self.model_input_mfcc_length}")
                continue

            # 计算当前20ms音频块的“瞬时”音量
            rms = np.sqrt(np.mean(np.square(new_samples_chunk)))
            rms_values.append(rms)

            # 转换为Tensor并进行推理
            features = torch.tensor(mfcc_matrix).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(features)
                probs = torch.softmax(outputs, dim=1)

            probabilities.append(probs.squeeze().numpy())
            current_time_sec = i / self.config.SR
            timestamps.append(current_time_sec)

            # --- 策略评估 ---
            heymemo_prob = probs.squeeze().numpy()[self.config.CLASS_NAMES_INDEX["HeyMemo"]]
            
            for strategy in self.word_strategies["HeyMemo"]["strategies"]:
                strategy.update_history(heymemo_prob, current_time_sec)
                triggered, trigger_type = strategy.check_trigger(current_time_sec)
                if triggered:
                    event_info = {
                        "time": current_time_sec,
                        "strategy_name": strategy.name,
                        "trigger_type": trigger_type
                    }
                    self.triggered_events.append(event_info)
                    print(f"Time: {current_time_sec:.2f}s - Strategy '{strategy.name}-{trigger_type}' would trigger.")

        return np.array(probabilities), np.array(timestamps), np.array(rms_values), self.triggered_events

class Visualizer:
    """
    可视化工具类，负责绘制图表。
    """
    def __init__(self, config):
        self.config = config

    def plot_probability_and_rms(self, output_path, title, probabilities, timestamps, rms_values, triggered_events):
        """
        绘制HeyMemo概率和音量图并保存到同一个图中（已优化触发点显示）。
        """
        if len(timestamps) == 0:
            print("没有数据可以绘制")
            return
        
        # 创建单个图表，使用双Y轴
        fig, ax1 = plt.subplots(figsize=(25, 12))
        
        # 将时间从秒转换为毫秒
        time_points_ms = [t * 1000 for t in timestamps]
        
        # 查找 "HeyMemo" 对应的索引
        heymemo_index = self.config.CLASS_NAMES_INDEX.get("HeyMemo")
        
        # 绘制HeyMemo概率曲线 (左Y轴)
        ax1.set_xlabel('Time(ms)')
        ax1.set_ylabel('HeyMemo Probability', color='black')
        if heymemo_index is not None:
            # 加粗概率曲线
            ax1.plot(time_points_ms, probabilities[:, heymemo_index], label='HeyMemo', linewidth=4.0, color='red')
        else:
            # 如果找不到HeyMemo，作为备用方案，绘制所有类别的概率
            for i, class_name in self.config.CLASS_NAMES.items():
                ax1.plot(time_points_ms, probabilities[:, i], label=f'Probability: {class_name}')

        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_ylim(0, 1)
        ax1.grid(True)
        
        # 创建共享X轴的第二个Y轴
        ax2 = ax1.twinx()
        
        # --- 音量处理 ---
        # 归一化处理
        volume_data_normalized = np.full_like(rms_values, np.nan, dtype=float)
        if rms_values.any() and rms_values.max() > 0:
            volume_data_normalized = rms_values / rms_values.max()

        # 绘制音量曲线 (右Y轴)
        ax2.set_ylabel('Volume', color='black')
        # 更改颜色为蓝色
        ax2.plot(time_points_ms, volume_data_normalized, 'b-', label='Volume', linewidth=2.0)
        ax2.tick_params(axis='y', labelcolor='black')
        # 调整Y轴范围
        ax2.set_ylim(0, 1.5)
        
        # --- 绘制策略触发点 (新版：点+线) ---
        strategy_styles = {
            "Peak":        {'color': 'purple', 'y': 0.85, 'marker': 'o'},
            "Combo":       {'color': 'green',  'y': 0.80, 'marker': 's'},
            "Average":     {'color': 'orange', 'y': 0.75, 'marker': 'D'},
            "Consecutive": {'color': 'cyan',   'y': 0.70, 'marker': '^'}
        }

        # 1. 绘制所有触发时间的垂直线 (去重)
        unique_trigger_times_ms = sorted(list(set(event['time'] * 1000 for event in triggered_events)))
        if unique_trigger_times_ms:
            # 只为第一条垂直线添加图例标签，避免重复
            ax1.axvline(x=unique_trigger_times_ms[0], color='gray', linestyle='--', linewidth=1.5, label='Trigger Time')
            for i in range(1, len(unique_trigger_times_ms)):
                ax1.axvline(x=unique_trigger_times_ms[i], color='gray', linestyle='--', linewidth=1.5)

        # 2. 按策略绘制散点，以区分重叠的触发
        plotted_strategy_labels = set()
        for event in triggered_events:
            time_ms = event["time"] * 1000
            strategy_name = event["strategy_name"]
            
            if strategy_name not in strategy_styles:
                continue
            
            style = strategy_styles[strategy_name]
            label = f'Trigger: {strategy_name}'
            
            # 为每种策略只添加一次图例
            if label not in plotted_strategy_labels:
                ax1.scatter(time_ms, style['y'], color=style['color'], marker=style['marker'], s=150, label=label, zorder=10)
                plotted_strategy_labels.add(label)
            else:
                ax1.scatter(time_ms, style['y'], color=style['color'], marker=style['marker'], s=150, zorder=10)

        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # 设置标题
        plt.title(title or 'HeyMemo Recognition and Volume')
        
        # 设置更紧密的时间轴刻度
        if time_points_ms:
            max_time = max(time_points_ms) if time_points_ms else 0
            plt.xticks(np.arange(0, max_time + 100, 200))
        
        plt.tight_layout()
        # 保存图像
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"图表已保存至: {output_path}")

# --- 主函数 ---
def run_visualization(model_path, wav_path, output_dir):
    """主运行函数"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化
    config = Config()
    simulator = StreamSimulator(model_path, config)
    visualizer = Visualizer(config)
    
    if os.path.isdir(wav_path):
        print(f"处理目录: {wav_path}")
        processed_files = 0
        triggered_files = 0
        strategy_counter = Counter()

        for root, _, files in os.walk(wav_path):
            wav_files = [f for f in files if f.endswith('.wav')]
            for file in wav_files:
                file_path = os.path.join(root, file)
                print(f"\n--- 处理文件: {file_path} ---")
                probabilities, timestamps, rms_values, triggered_events = simulator.process_audio_stream(file_path)
                
                if len(probabilities) > 0:
                    file_name = os.path.basename(file_path)
                    output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_prob_dist.png")
                    visualizer.plot_probability_and_rms(output_path, f"{file_name} Probability & Volume", probabilities, timestamps, rms_values, triggered_events)
                
                processed_files += 1
                if triggered_events:
                    triggered_files += 1
                    # 每个文件只统计一次触发的策略类型，避免重复计数
                    unique_strategies_in_file = {event['strategy_name'] for event in triggered_events}
                    strategy_counter.update(unique_strategies_in_file)

        # --- 生成、打印并保存统计结果 ---
        summary_lines = []
        summary_lines.append(f"共处理了 {processed_files} 个音频文件")

        wake_rate = (triggered_files / processed_files * 100) if processed_files > 0 else 0
        summary_lines.append(f"\n--- 唤醒统计结果 ---")
        summary_lines.append(f"成功唤醒文件数: {triggered_files}/{processed_files}")
        summary_lines.append(f"唤醒率: {wake_rate:.2f}%")
        
        summary_lines.append(f"\n--- 各类唤醒策略统计 ---")
        if triggered_files > 0:
            # 按触发次数降序排序
            sorted_strategies = sorted(strategy_counter.items(), key=lambda item: item[1], reverse=True)
            for strategy, count in sorted_strategies:
                # 此处的百分比表示该策略在所有被唤醒的文件中的贡献度
                strategy_rate = (count / triggered_files * 100)
                summary_lines.append(f"{strategy}: {count}次 ({strategy_rate:.2f}%)")
        else:
            summary_lines.append("没有任何文件被唤醒。")

        # 在控制台打印结果
        print("\n" + "\n".join(summary_lines))

        # 保存结果到文件
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        print(f"\n统计结果已保存至: {summary_path}")

    else:
        print(f"--- 处理文件: {wav_path} ---")
        probabilities, timestamps, rms_values, triggered_events = simulator.process_audio_stream(wav_path)
        if len(probabilities) > 0:
            file_name = os.path.basename(wav_path)
            output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_prob_dist.png")
            visualizer.plot_probability_and_rms(output_path, f"{file_name} Probability & Volume", probabilities, timestamps, rms_values, triggered_events)

if __name__ == "__main__":
    # 请根据实际情况修改模型和数据路
    model_path = 'checkpoint_2.2_ft2/crnn_model_best.pth'
    data_path = './string_sample/edgetts_test.wav'
    output_directory = './string_plots/pos'
    
    run_visualization(model_path, data_path, output_directory)
