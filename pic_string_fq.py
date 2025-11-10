import re
import os
import sys
import math
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
QUANT_LOG_PATH = "./quantproject_2.2_ft3/recording2.log" # 量化模型log
USE_QUANT_LOG = True # 是否使用量化的log
INPUT_AUDIO_PATH = "./string_sample/other/recording_20251021_103943_240.wav" # 输入的wav文件
OUTPUT_DIR = "./test_results/recording_0.999_0.999/" # 输出路径
RELATIVE_TIME = True
DPI = 60
MODEL_PATH = "checkpoint/checkpoint_2.2_ft3/crnn_model_best.pth" # 模型路径
ORIGINAL_LOG_PATH = os.path.join(OUTPUT_DIR, "float_recording.log") # 原始模型log存储位置
MERGED_LOG_PATH_TEMPLATE = os.path.join(OUTPUT_DIR, "merged_recording.log") # 合并后的log存储位置
OUTPUT_PLOT_PATH = os.path.join(OUTPUT_DIR, "merged_recording.png") # 合并后概率图像存储位置
SAVE_ORIGINAL_LOG = True # 是否存储原始模型log
SAVE_MERGED_LOG = True # 是否存储合并后的log
SAVE_PLOT = True # 是否存储合并后的概率图像
PLOT_SEGMENT_SECONDS = 60  # 每张子图的时间长度（秒）

def sanitize_filename(name: str) -> str:
    # Windows 文件名非法字符替换为下划线
    return re.sub(r'[\\/:\*\?"<>\|.]', '', name)

def make_segment_plot_path(base_path: str, index: int) -> str:
    # 生成分段图像路径：base名 + _index + .png
    root, _ = os.path.splitext(base_path)
    return f"{root}_{index}.png"

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

        # 统一一次性打开 original.log，必要时创建目录（可选）
        original_log_file = None
        if SAVE_ORIGINAL_LOG:
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
                if original_log_file:
                    original_log_file.write(
                        f"time: {frame_time_sec:.2f}s outputs: {outputs_str} probs: {probs_str}\n"
                    )
                timestamps.append(frame_time_sec)
                heymemo_probs.append(float(probs[self.config.CLASS_NAMES_INDEX['HeyMemo']]))

                # 生成合并日志行（仅当启用且匹配到量化数据）
                if generate_merged_log and quant_data:
                    time_key = f"{frame_time_sec:.2f}"
                    if time_key in quant_data:
                        quant_info = quant_data[time_key]
                        merged_line = (
                            f"time:{frame_time_sec:.2f} "
                            f"float:{outputs_str} "
                            f"probs:{probs_str} "
                            # f"quant: class:{quant_info['class']}, "
                            # f"idx:{quant_info['idx']}, "
                            f"float_net:[{quant_info['float_net']}], "
                            f"softmax_result:[{quant_info['softmax_result']}]"
                        )
                        merged_log_lines.append(merged_line)

        finally:
            if original_log_file:
                original_log_file.close()

        # 归一化音量
        rms_values = np.array(rms_values, dtype=float)
        if rms_values.size > 0 and np.max(rms_values) > 0:
            rms_values = rms_values / np.max(rms_values)
        
        if generate_merged_log:
            return timestamps, heymemo_probs, np.array(rms_values), merged_log_lines
        return timestamps, heymemo_probs, np.array(rms_values)

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
    softmax_bracket_re_old = re.compile(r'softmax_result:\[(.*?)\]')
    softmax_bracket_re_new = re.compile(r'softmax:\s*\[(.*?)\]')

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 兼容两种 softmax 写法
            t_match = time_re.search(line)
            if not t_match:
                continue
            s_match = softmax_bracket_re_old.search(line) or softmax_bracket_re_new.search(line)
            if not s_match:
                continue

            # 提取数值
            s_content = s_match.group(1)
            nums = re.findall(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?', s_content)
            if len(nums) < 2:
                continue
            p0 = float(nums[0]); p1 = float(nums[1])

            # 旧日志有“文件名, time:...”，新日志没有文件名；使用默认键
            parts = line.split(',', 1)
            if len(parts) >= 2 and 'time:' in parts[1]:
                file_id = parts[0].strip()
            else:
                file_id = '__GLOBAL__'

            if file_id not in data:
                data[file_id] = {"times": [], "p0": [], "p1": []}
            data[file_id]["times"].append(float(t_match.group(1)))
            data[file_id]["p0"].append(p0)
            data[file_id]["p1"].append(p1)
    return data

def parse_quant_data_for_merge(file_path):
    """解析量化日志，返回按时间索引的字典，用于合并日志生成"""
    data = {}
    time_re = re.compile(r'time:(\d+(?:\.\d+)?)s')
    logits_re = re.compile(r'\[\[\s*(.*?)\s*\]\]')           # 匹配 [[ logits ]]
    softmax_re = re.compile(r'softmax:\s*\[\s*(.*?)\s*\]')   # 匹配 softmax: [ ... ]

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            t_match = time_re.search(line)
            if not t_match:
                continue
            sm_match = softmax_re.search(line)
            if not sm_match:
                # 兼容旧格式 softmax_result:[...]
                sm_match = re.search(r'softmax_result:\[(.*?)\]', line)
                if not sm_match:
                    continue

            time_key = f"{float(t_match.group(1)):.2f}"
            logits_match = logits_re.search(line)
            logits_content = logits_match.group(1).strip() if logits_match else ""
            softmax_content = sm_match.group(1).strip()

            # 填充必要字段，旧代码的合并逻辑可继续运行
            data[time_key] = {
                'float_net': logits_content,  # 用 [[ logits ]] 作为“float_net”的内容
                'softmax_result': softmax_content,
            }
    return data

def main():
    grouped = parse_log_group_by_file(QUANT_LOG_PATH)
    if not grouped:
        print(f"没有解析到任何有效数据，请检查日志文件: {QUANT_LOG_PATH}")
        return

    # 不再强制依赖量化日志分组解析，允许无量化数据运行
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 准备一次性加载的流式推理器
    sim = StreamSimulator(MODEL_PATH, Config())

    # 可选解析量化数据（用于画量化曲线和可选的合并日志）
    quant_data = None
    times, p1_list = [], []
    if USE_QUANT_LOG and os.path.isfile(QUANT_LOG_PATH):
        quant_data = parse_quant_data_for_merge(QUANT_LOG_PATH)
        order = sorted(quant_data.keys(), key=lambda k: float(k))
        times = [float(k) for k in order]
        for k in order:
            s_content = quant_data[k]['softmax_result']
            nums = re.findall(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?', s_content)
            if len(nums) >= 2:
                p1_list.append(float(nums[1]))

    # 直接使用顶部常量定义的音频路径进行流式推理
    wav_path = INPUT_AUDIO_PATH
    pic_times, pic_probs, pic_rms = [], [], np.array([])
    merged_log_lines = []

    result = sim.run_once(
        wav_path,
        generate_merged_log=SAVE_MERGED_LOG,
        quant_data=quant_data
    )
    if SAVE_MERGED_LOG:
        pic_times, pic_probs, pic_rms, merged_log_lines = result
        # 使用模板生成合并日志路径；支持无占位的固定路径
        file_stub = sanitize_filename(os.path.basename(wav_path))
        if "{file_stub}" in MERGED_LOG_PATH_TEMPLATE:
            merged_log_path = MERGED_LOG_PATH_TEMPLATE.format(file_stub=file_stub)
        else:
            merged_log_path = MERGED_LOG_PATH_TEMPLATE

        # 写入前确保目录存在（仅当需要保存且有内容）
        if merged_log_lines:
            merged_dir = os.path.dirname(merged_log_path)
            if merged_dir and not os.path.isdir(merged_dir):
                os.makedirs(merged_dir, exist_ok=True)
            with open(merged_log_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(merged_log_lines))
            print(f"已生成合并日志: {merged_log_path}", file=sys.stderr)
    else:
        pic_times, pic_probs, pic_rms = result

    # 相对时间对齐（两条曲线各自从 0s 开始）
    if RELATIVE_TIME:
        if len(times) > 0:
            t0 = times[0]; times = [t - t0 for t in times]
        if len(pic_times) > 0:
            t1 = pic_times[0]; pic_times = [t - t1 for t in pic_times]

    if SAVE_PLOT:
        # 计算总时长与分段数量
        total_max_t = max([max(times) if times else 0.0,
                           max(pic_times) if pic_times else 0.0])
        n_segments = max(1, math.ceil(total_max_t / PLOT_SEGMENT_SECONDS))

        def slice_series(xs, ys, start, end):
            seg_x, seg_y = [], []
            for x, y in zip(xs, ys):
                if start <= x < end:
                    seg_x.append(x - start)  # 每段内从0开始
                    seg_y.append(y)
            return seg_x, seg_y

        def slice_series_np(xs, ys_np, start, end):
            seg_x, seg_y = [], []
            for i, x in enumerate(xs):
                if start <= x < end:
                    seg_x.append(x - start)
                    seg_y.append(float(ys_np[i]))
            return seg_x, seg_y

        # 逐段创建并保存独立的 PNG
        for i in range(n_segments):
            seg_start = i * PLOT_SEGMENT_SECONDS
            seg_end = (i + 1) * PLOT_SEGMENT_SECONDS

            plt.figure(figsize=(30, 8), dpi=DPI)
            ax = plt.gca()
            ax.grid(True, linestyle="--", alpha=0.4)

            # 橙色：量化 softmax[1]（仅在启用且有数据）
            if times and p1_list and USE_QUANT_LOG:
                seg_x_q, seg_y_q = slice_series(times, p1_list, seg_start, seg_end)
                if seg_x_q:
                    ax.plot(seg_x_q, seg_y_q, label="quant softmax[1]", color="orange", lw=3)

            # 红色：浮点 softmax[1]
            if pic_times and pic_probs:
                seg_x_f, seg_y_f = slice_series(pic_times, pic_probs, seg_start, seg_end)
                if seg_x_f:
                    ax.plot(seg_x_f, seg_y_f, label="float softmax[1]", color="red", lw=3)

            # 蓝色：音量（归一化）
            if pic_times and pic_rms.size > 0:
                seg_x_v, seg_y_v = slice_series_np(pic_times, 0.5 * pic_rms, seg_start, seg_end)
                if seg_x_v:
                    ax.plot(seg_x_v, seg_y_v, label="volume", color="blue", lw=1)

            # 该段的 x 轴刻度：每秒一个
            seg_len = min(PLOT_SEGMENT_SECONDS, max(0.0, total_max_t - seg_start))
            ax.set_xticks(list(range(0, int(math.ceil(seg_len)) + 1, 1)))
            ax.set_xlim(0, seg_len)

            ax.set_xlabel("Time(s)" if RELATIVE_TIME else "Time(s, absolute)")
            ax.set_ylabel("Value")
            ax.set_title(f"{os.path.basename(INPUT_AUDIO_PATH)} — {seg_start:.0f}s to {min(seg_end, total_max_t):.0f}s")
            ax.legend()

            segment_path = make_segment_plot_path(OUTPUT_PLOT_PATH, i + 1)
            # 保存前确保分段图像所在目录存在
            segment_dir = os.path.dirname(segment_path)
            if segment_dir and not os.path.isdir(segment_dir):
                os.makedirs(segment_dir, exist_ok=True)

            plt.tight_layout()
            plt.savefig(segment_path)
            plt.close()
            print(f"已保存图像到: {segment_path}")
    else:
        print("已跳过绘图（SAVE_PLOT=False）")

if __name__ == "__main__":
    main()
