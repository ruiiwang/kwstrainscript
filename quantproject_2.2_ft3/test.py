import os
import sys
import io
import re
import librosa
import torch
import json5
import numpy as np
from quantize_public.batch_gen_quant_data import run_net
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# 多进程配置
USE_MULTIPROCESS = True
NUM_WORKERS = max(1, (os.cpu_count() or 1) - 1)

# 类名映射
CLASS_NAMES = {
    0: "UNKNOWN_WORD",
    1: "HeyMemo",
}

# 模型和配置文件路径
CHECKPOINT_PATH = './entercompany_checkpoint/'
CONFIG_FILE = os.path.join(CHECKPOINT_PATH, "config_2classes_20250909.json5")
ORIGINAL_CHECKPOINT_NAME = 'crnn_model_best.pth'
QUANTIZED_MODEL_FILE = os.path.join(CHECKPOINT_PATH, 'crnn_model_best_quant.pt')
QUANT_PARAMS_FILE = os.path.join(CHECKPOINT_PATH, 'crnn_model_best_quant_params_Q.txt')

def softmax(x):
    """计算 softmax"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def process_audio_segment_core(segment, timestamp, audio_path, checkpoint_path, quant_params_file, 
                              model_config, feature_config, quantized_model_file):
    """处理单个音频片段的核心逻辑"""
    captured_output = io.StringIO()
    original_stdout = sys.stdout
    try:
        sys.stdout = captured_output
        prediction = run_net(segment, checkpoint_path, quant_params_file, 
                           model_config, feature_config, quantized_model_file, 0)
        output_text = captured_output.getvalue()
    finally:
        sys.stdout = original_stdout

    fixed_net = None
    float_net = None
    bracket_matches = re.findall(r'\[([^\]]+)\]', output_text)
    if len(bracket_matches) >= 2:
        fixed_net = bracket_matches[0].strip()
        float_net = bracket_matches[1].strip()
        try:
            numbers = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', float_net)
            extracted_array = np.array([float(n) for n in numbers])
        except:
            extracted_array = None
    else:
        print(f"未找到足够的括号数组: {output_text}")
        extracted_array = None

    softmax_result = None
    if extracted_array is not None:
        softmax_result = softmax(extracted_array)

    if isinstance(prediction, torch.Tensor):
        pred_idx = prediction.item() if prediction.numel() == 1 else prediction.argmax().item()
    else:
        pred_idx = prediction if isinstance(prediction, int) else int(prediction)
    predicted_class = CLASS_NAMES.get(pred_idx, f"未知类别 ({pred_idx})")

    return (
        f"{audio_path}, time:{timestamp:.2f}s, class:{predicted_class}, "
        f"idx:{pred_idx}, "
        f"fixed_net:[{fixed_net}], "
        f"float_net:[{float_net}], "
        f"softmax_result:{softmax_result.tolist() if softmax_result is not None else 'None'}\n"
    )


def process_audio_segment(args):
    """多进程处理单个音频片段的包装函数"""
    return process_audio_segment_core(*args)

def run_stream_inference_on_folder(input_dir, log_file):
    required_files = [CONFIG_FILE, QUANTIZED_MODEL_FILE, QUANT_PARAMS_FILE]
    for f in required_files:
        if not os.path.exists(f):
            print(f"错误：在 '{f}' 处找不到必需的文件")
            return

    with open(CONFIG_FILE, 'r') as f:
        config = json5.load(f)
    model_config = config["feature"]
    feature_config = config["feature_input"]

    sr = 16000
    window_length = 1.6
    hop_length = 0.02
    window_samples = int(window_length * sr)
    hop_samples = int(hop_length * sr)

    with open(log_file, 'w') as logf:
        def process_file(audio_path: str):
            try:
                signal, _ = librosa.core.load(audio_path, sr=sr)
            except Exception as e:
                print(f"加载音频失败: {audio_path}, 错误: {e}")
                return
            print(f"处理文件: {audio_path}")
            
            # 在信号前拼接空白音频作为缓冲区
            buffer = np.zeros(window_samples, dtype=np.float32)
            processed_signal = np.concatenate((buffer, signal))
            
            # 根据片段数量选择处理方式
            if USE_MULTIPROCESS and len(processed_signal) > window_samples * 10:
                # 多进程处理
                segment_args = []
                for i in range(0, len(processed_signal) - window_samples + 1, hop_samples):
                    segment = processed_signal[i:i+window_samples]
                    timestamp = i / sr
                    args = (segment, timestamp, audio_path, CHECKPOINT_PATH, QUANT_PARAMS_FILE,
                           model_config, feature_config, QUANTIZED_MODEL_FILE)
                    segment_args.append(args)
                
                print(f"使用 {NUM_WORKERS} 个进程处理 {len(segment_args)} 个音频片段")
                with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                    results = list(tqdm(executor.map(process_audio_segment, segment_args), 
                                      total=len(segment_args), desc="处理音频片段"))
                
                for log_line in results:
                    logf.write(log_line)
                    logf.flush()
            else:
                # 单进程处理
                for i in tqdm(range(0, len(processed_signal) - window_samples + 1, hop_samples)):
                    segment = processed_signal[i:i+window_samples]
                    timestamp = i / sr
                    log_line = process_audio_segment_core(segment, timestamp, audio_path, CHECKPOINT_PATH, 
                                                        QUANT_PARAMS_FILE, model_config, feature_config, QUANTIZED_MODEL_FILE)
                    logf.write(log_line)
                    logf.flush()

        # 处理单文件或目录
        if os.path.isfile(input_dir):
            if input_dir.lower().endswith('.wav'):
                process_file(input_dir)
            else:
                print(f"错误：不支持的文件类型（仅支持 .wav）: {input_dir}")
        else:
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith('.wav'):
                        process_file(os.path.join(root, file))

    print(f"所有推理结果已保存到 {log_file}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    # input_dir = "/mnt/f/realtime"
    input_dir = "../string_sample/other/recording_20251021_103943_240.wav"
    # input_dir = "/mnt/c/Users/Win11/Downloads/CHiME6"
    log_file = "./recording.log"
    run_stream_inference_on_folder(input_dir, log_file)
