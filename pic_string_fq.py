import re
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import librosa
from collections import deque
from model.crnn_model import CnnRnnModel1Channel
from mfcc_io import mfcc

'''
代码功能：
    从量化后模型测试工程生成的 .log 中提取 quant: class:, idx:, float_net[], softmax[] 信息；
    流式仿真，使用相同的 .wav 对量化前模型进行推理，生成 timestamp, float[], probs(softmax)[] 信息；
    生成格式如 time:, float[], probs[], quant: class:, idx:, float_net:[], softmax_result:[] 的 .log；
    绘制量化前、量化后 HeyMemo 类别的概率变化图（附加音量），保存为 .png。
'''
# 顶部常量区域
LOG_PATH = r"./quantproject_2.2_ft2/heymemo_short.log"
OUTPUT_DIR = r"./test_results"
RELATIVE_TIME = True
DPI = 60
MODEL_PATH = r"checkpoint/checkpoint_2.2_ft2/crnn_model_best.pth"
ORIGINAL_LOG_PATH = r"./float_heymemo_short.log"
MERGED_LOG_PATH_TEMPLATE = r"./merged_heymemo_short.log"
OUTPUT_PLOT_PATH = r"./test_results/merged_heymemo_short.png"

def sanitize_filename(name: str) -> str:
    # Windows 文件名非法字符替换为下划线
    return re.sub(r'[\\/:\*\?"<>\|.]', '', name)

class Config:
    SR = 16000
    WINDOW_LENGTH_S = 1.6
    HOP_LENGTH_S = 0.02
    AUDIO_BUFFER_SIZE = int(WINDOW_LENGTH_S * SR)
    HOP_SAMPLES = int(HOP_LENGTH_S * SR)
    MFCC_HOP_LENGTH = 256
    MFCC_WIN_LENGTH = 512
    CLASS_NAMES = {0: "UNKNOWN_WORD", 1: "HeyMemo"}
    CLASS_NAMES_INDEX = {name: index for index, name in CLASS_NAMES.items()}
    MODEL_CONFIG = {
        "in_c": 16,
        "conv": [{"out_c": 16, "k": 8, "s": 2, "p": 1, "dropout": 0.0},
                 {"out_c": 32, "k": 4, "s": 2, "p": 1, "dropout": 0.0}],
        "rnn": {"dim": 32, "layers": 1, "dropout": 0.2, "bidirectional": True},
        "fc_out": len(CLASS_NAMES),
    }

class StreamSimulator:
    def __init__(self, model_path, config):
        self.config = config
        self.model = self._load_model(model_path)
        self.model_input_mfcc_length = self._calculate_mfcc_length()

    def _load_model(self, model_path):
        model = CnnRnnModel1Channel(self.config.MODEL_CONFIG)
        try:
            model.load_state_dict(torch.load(model_path))
        except KeyError:
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def _get_mfcc_from_audio(self, audio_data):
        audio_f32 = np.asarray(audio_data, dtype=np.float32)
        return mfcc(y=audio_f32, sr=self.config.SR, n_mfcc=16, n_mels=40,
                    win_length=512, window='hamming',
                    hop_length=self.config.MFCC_HOP_LENGTH, n_fft=512,
                    fmin=20, fmax=4050)

    def _calculate_mfcc_length(self):
        zero_audio = np.zeros(self.config.AUDIO_BUFFER_SIZE, dtype=np.float32)
        mfccs = self._get_mfcc_from_audio(zero_audio)
        return mfccs.shape[1]

    def run_once(self, wav_path, generate_merged_log=False, quant_data=None):
        # 返回：timestamps(s), heymemo_prob(list), rms(list) 或 (timestamps, heymemo_prob, rms, merged_log_lines)
        audio, _ = librosa.load(wav_path, sr=self.config.SR)
        audio_buffer = deque([0.0] * self.config.AUDIO_BUFFER_SIZE, maxlen=self.config.AUDIO_BUFFER_SIZE)
        buffer_array = np.array(audio_buffer, dtype=np.float32)
        mfcc_matrix = self._get_mfcc_from_audio(buffer_array)
        unprocessed_audio = []

        timestamps, heymemo_probs, rms_values = [], [], []
        merged_log_lines = []

        # 统一一次性打开 original.log，必要时创建目录
        orig_dir = os.path.dirname(ORIGINAL_LOG_PATH)
        if orig_dir and not os.path.isdir(orig_dir):
            os.makedirs(orig_dir, exist_ok=True)
        original_log_file = open(ORIGINAL_LOG_PATH, "a", encoding="utf-8")

        try:
            for i in range(0, len(audio), self.config.HOP_SAMPLES):
                new_samples_chunk = audio[i: i + self.config.HOP_SAMPLES]
                unprocessed_audio.extend(new_samples_chunk)
                num_frames_to_update = len(unprocessed_audio) // self.config.MFCC_HOP_LENGTH
                if num_frames_to_update > 0:
                    for _ in range(num_frames_to_update):
                        frame_to_process = unprocessed_audio[:self.config.MFCC_HOP_LENGTH]
                        audio_buffer.extend(frame_to_process)
                        buffer_array = np.array(audio_buffer, dtype=np.float32)
                        new_frame_audio = buffer_array[-512:]
                        new_mfcc_frames = self._get_mfcc_from_audio(new_frame_audio)
                        new_mfcc_column = new_mfcc_frames[:, -1]
                        mfcc_matrix = np.roll(mfcc_matrix, -1, axis=1)
                        mfcc_matrix[:, -1] = new_mfcc_column
                    processed_samples = num_frames_to_update * self.config.MFCC_HOP_LENGTH
                    unprocessed_audio = unprocessed_audio[processed_samples:]

                if mfcc_matrix.shape[1] != self.model_input_mfcc_length:
                    continue

                rms = np.sqrt(np.mean(np.square(new_samples_chunk)))
                rms_values.append(rms)

                # 模型输入：float32 + 形状 (1, in_c, time)
                features = torch.tensor(mfcc_matrix, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    outputs = self.model(features)                      # shape: (1, fc_out)
                    probs_tensor = torch.softmax(outputs, dim=-1)[0]   # shape: (fc_out,)
                    probs = probs_tensor.cpu().numpy()

                frame_time_sec = i / self.config.SR
                outputs_str = f"[{outputs[0, 0].item():.6f}, {outputs[0, 1].item():.6f}]"
                probs_str = f"[{probs[0]:.6f}, {probs[1]:.6f}]"
                original_log_file.write(
                    f"time: {frame_time_sec:.2f}s outputs: {outputs_str} probs: {probs_str}\n"
                )
                timestamps.append(frame_time_sec)
                heymemo_probs.append(float(probs[self.config.CLASS_NAMES_INDEX["HeyMemo"]]))

                # 生成合并日志行
                if generate_merged_log and quant_data:
                    time_key = f"{frame_time_sec:.2f}"
                    if time_key in quant_data:
                        quant_info = quant_data[time_key]
                        merged_line = (
                            f"time:{frame_time_sec:.2f} "
                            f"float:{outputs_str} "
                            f"probs:{probs_str} "
                            f"quant: class:{quant_info['class']}, "
                            f"idx:{quant_info['idx']}, "
                            f"float_net:[{quant_info['float_net']}], "
                            f"softmax_result:[{quant_info['softmax_result']}]"
                        )
                        merged_log_lines.append(merged_line)

        finally:
            original_log_file.close()
            
        # 归一化音量
        rms_values = np.array(rms_values, dtype=float)
        if rms_values.size > 0 and np.max(rms_values) > 0:
            rms_values = rms_values / np.max(rms_values)
        
        if generate_merged_log:
            return timestamps, heymemo_probs, np.array(rms_values), merged_log_lines
        return timestamps, heymemo_probs, np.array(rms_values)

# 新增：根据日志中的首字段推断本地可用的 wav 路径
_FIND_CACHE = {}
def find_audio_path(file_id: str, log_path: str):
    if file_id in _FIND_CACHE:
        return _FIND_CACHE[file_id]
    # 1) 原样可访问
    if os.path.isfile(file_id):
        _FIND_CACHE[file_id] = file_id
        return file_id
    # 2) 相对于日志目录解析（处理 "../xxx.wav"）
    base_dir = os.path.dirname(os.path.abspath(log_path))
    candidate = os.path.normpath(os.path.join(base_dir, file_id))
    if os.path.isfile(candidate):
        _FIND_CACHE[file_id] = candidate
        return candidate
    # 3) 按文件名在当前工程下搜寻
    target = os.path.basename(file_id.replace('\\', '/'))
    for root, _, files in os.walk('.'):
        for f in files:
            if f.lower() == target.lower():
                path = os.path.join(root, f)
                _FIND_CACHE[file_id] = path
                return path
    _FIND_CACHE[file_id] = None
    return None

def parse_log_group_by_file(file_path):
    data = {}  # {file_id: {"times":[], "p0":[], "p1":[]}}
    time_re = re.compile(r'time:(\d+(?:\.\d+)?)s')
    softmax_bracket_re = re.compile(r'softmax_result:\[(.*?)\]')

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            file_id = line.split(',', 1)[0].strip()
            t_match = time_re.search(line)
            if not t_match or 'softmax_result:None' in line:
                continue
            s_match = softmax_bracket_re.search(line)
            if not s_match:
                continue
            t = float(t_match.group(1))
            s_content = s_match.group(1)
            nums = re.findall(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?', s_content)
            if len(nums) < 2:
                continue
            p0 = float(nums[0]); p1 = float(nums[1])
            if file_id not in data:
                data[file_id] = {"times": [], "p0": [], "p1": []}
            data[file_id]["times"].append(t)
            data[file_id]["p0"].append(p0)
            data[file_id]["p1"].append(p1)
    return data

def parse_quant_data_for_merge(file_path):
    """解析量化日志，返回按时间索引的字典，用于合并日志生成"""
    data = {}
    time_re = re.compile(r'time:(\d+(?:\.\d+)?)s')
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            t_match = time_re.search(line)
            if not t_match:
                continue
            
            # 提取各字段
            m_cls = re.search(r'class:([^,]+)', line)
            m_idx = re.search(r'idx:(\d+)', line)
            m_float_net = re.search(r'float_net:\[(.*?)\]', line)
            m_softmax = re.search(r'softmax_result:\[(.*?)\]', line)
            
            if not (m_cls and m_idx and m_float_net and m_softmax):
                continue
            
            time_key = f"{float(t_match.group(1)):.2f}"
            data[time_key] = {
                'class': m_cls.group(1).strip(),
                'idx': m_idx.group(1).strip(),
                'float_net': m_float_net.group(1).strip(),
                'softmax_result': m_softmax.group(1).strip(),
            }
    return data

def main():
    grouped = parse_log_group_by_file(LOG_PATH)
    if not grouped:
        print(f"没有解析到任何有效数据，请检查日志文件: {LOG_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 新增：准备一次性加载的流式推理器
    sim = StreamSimulator(MODEL_PATH, Config())
    
    # 新增：解析量化数据用于合并日志
    quant_data = parse_quant_data_for_merge(LOG_PATH)

    for file_id, series in grouped.items():
        times = series["times"]; p0_list = series["p0"]; p1_list = series["p1"]
        if not times:
            continue

        # 排序
        order = sorted(range(len(times)), key=lambda i: times[i])
        times = [times[i] for i in order]
        p0_list = [p0_list[i] for i in order]
        p1_list = [p1_list[i] for i in order]

        # 解析本地 wav 路径并跑一次 pic_string 流式
        wav_path = find_audio_path(file_id, LOG_PATH)
        pic_times, pic_probs, pic_rms = [], [], np.array([])
        merged_log_lines = []
        
        if wav_path and os.path.isfile(wav_path):
            result = sim.run_once(wav_path, generate_merged_log=True, quant_data=quant_data)
            pic_times, pic_probs, pic_rms, merged_log_lines = result

            # 使用模板生成合并日志路径；支持无占位的固定路径
            file_stub = sanitize_filename(file_id)
            if "{file_stub}" in MERGED_LOG_PATH_TEMPLATE:
                merged_log_path = MERGED_LOG_PATH_TEMPLATE.format(file_stub=file_stub)
            else:
                merged_log_path = MERGED_LOG_PATH_TEMPLATE

            # 写入前确保目录存在
            merged_dir = os.path.dirname(merged_log_path)
            if merged_dir and not os.path.isdir(merged_dir):
                os.makedirs(merged_dir, exist_ok=True)

            if merged_log_lines:
                with open(merged_log_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(merged_log_lines))
                print(f"已生成合并日志: {merged_log_path}", file=sys.stderr)
        else:
            # 不要在没有路径时调用推理
            print(f"警告：未找到对应音频文件，跳过 pic 曲线: {file_id}")
            pic_times, pic_probs, pic_rms = sim.run_once(wav_path)

        # 相对时间对齐（两条曲线各自从 0s 开始）
        if RELATIVE_TIME:
            if len(times) > 0:
                t0 = times[0]; times = [t - t0 for t in times]
            if len(pic_times) > 0:
                t1 = pic_times[0]; pic_times = [t - t1 for t in pic_times]

        plt.figure(figsize=(30, 15), dpi=DPI)
        # 橙色：日志 softmax[1]（量化链路）
        plt.plot(times, p1_list, label="quant softmax[1]", color="orange", lw=3)
        # 红色：pic_string softmax[1]（浮点模型链路）
        if pic_times:
            plt.plot(pic_times, pic_probs, label="float softmax[1]", color="red", lw=3)
        # 蓝色：音量（归一化）
        if pic_times and pic_rms.size > 0:
            plt.plot(pic_times, 0.5*pic_rms, label="volume", color="blue", lw=1)

        # x 轴每秒一个刻度
        ax = plt.gca()
        max_t = max([max(times) if times else 0.0, max(pic_times) if pic_times else 0.0])
        ax.set_xticks(list(range(0, int(max_t) + 1, 1)))

        plt.xlabel("Time(s)" if RELATIVE_TIME else "Time(s, absolute)")
        plt.ylabel("Value")
        plt.title(f"Log vs Pic Softmax[1] + Volume - {file_id}")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT_PATH)
        plt.close()
        print(f"已保存图像到: {OUTPUT_PLOT_PATH}")

if __name__ == "__main__":
    main()