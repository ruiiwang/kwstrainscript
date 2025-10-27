import os
import sys
import io
import re
import librosa
import torch
import json5
import numpy as np
from quantize_public.batch_gen_quant_data import run_net

# 添加类名映射
CLASS_NAMES = {
    0: "UNKNOWN_WORD",
    1: "HeyMemo",
}

# 新增：定义 softmax 函数
def softmax(x):
    """
    对输入数组进行 softmax 运算。
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def run_stream_inference_on_folder(input_dir, log_file='./realtime.log'):
    checkpoint_path = './entercompany_checkpoint/'
    # 使用单类别配置文件
    config_file = os.path.join(checkpoint_path, "config_1class_20251017.json5")
    
    original_checkpoint_name = 'crnn_model_best.pth'
    
    quantized_model_name = original_checkpoint_name.split('.')[0] + '_quant.pt'
    quantized_model_file = os.path.join(checkpoint_path, quantized_model_name)
    quant_params_file = os.path.join(checkpoint_path, original_checkpoint_name.split('.')[0] + '_quant_params_Q' + '.txt')

    required_files = [config_file, quantized_model_file, quant_params_file]
    for f in required_files:
        if not os.path.exists(f):
            print(f"错误：在 '{f}' 处找不到必需的文件")
            return

    # 1. 加载模型和特征配置
    with open(config_file, 'r') as f:
        config = json5.load(f)
    model_config = config["feature"]
    feature_config = config["feature_input"]
    # 正类索引（默认 HeyMemo=1）
    target_class = config.get("classes", {})
    pos_class_idx = int(target_class.get("HeyMemo", 1))

    sr = 16000
    window_length = 1.6
    hop_length = 0.2
    window_samples = int(window_length * sr)
    hop_samples = int(hop_length * sr)
    buffer_samples = window_samples

    # 阈值：最终分类以该阈值为准（可自行调整）
    threshold = 0.9

    with open(log_file, 'w') as logf:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith('.wav'):
                    audio_path = os.path.join(root, file)
                    try:
                        signal, _ = librosa.core.load(audio_path, sr=sr)
                    except Exception as e:
                        print(f"加载音频失败: {audio_path}, 错误: {e}")
                        continue
                    print(f"处理文件: {audio_path}")
                    
                    # --- 新增逻辑：在信号前拼接空白音频作为缓冲区 ---
                    # 创建一个全零的缓冲区
                    buffer = np.zeros(buffer_samples, dtype=np.float32)
                    # 将缓冲区和原始信号拼接在一起
                    processed_signal = np.concatenate((buffer, signal))
                    
                    # 循环处理新的拼接信号
                    # 注意：循环的范围从 0 开始，总长度是拼接后的信号长度
                    for i in range(0, len(processed_signal) - window_samples + 1, hop_samples):
                        segment = processed_signal[i:i+window_samples]
                        timestamp = (i - buffer_samples) / sr

                        # 移除非负时间戳过滤，保留负时间段输出
                        captured_output = io.StringIO()
                        original_stdout = sys.stdout
                        try:
                            sys.stdout = captured_output
                            prediction = run_net(segment, checkpoint_path, quant_params_file, model_config, feature_config, quantized_model_file, pos_class_idx)
                            output_text = captured_output.getvalue()
                        finally:
                            sys.stdout = original_stdout

                        # 解析输出，兼容单值或多值数组
                        extracted_array = None
                        match = re.search(r'\[\s*(?:-?\d+\.\d+\s*)+\]', output_text)
                        if match:
                            data_string = match.group(0)
                            numbers = re.findall(r'-?\d+\.\d+', data_string)
                            extracted_array = np.array([float(n) for n in numbers], dtype=np.float32)

                        # 计算正类概率：
                        # - 单输出：sigmoid(logit)
                        # - 多输出：softmax(logits)[pos_class_idx]
                        positive_prob = None
                        if extracted_array is not None:
                            if extracted_array.size == 1:
                                positive_prob = float(1.0 / (1.0 + np.exp(-extracted_array[0])))
                            else:
                                sm = softmax(extracted_array)
                                if pos_class_idx < sm.size:
                                    positive_prob = float(sm[pos_class_idx])

                        # 按阈值进行最终分类
                        if positive_prob is not None and positive_prob >= threshold:
                            pred_idx = 1
                        else:
                            pred_idx = 0
                        predicted_class = CLASS_NAMES.get(pred_idx, f"未知类别 ({pred_idx})")

                        # 记录日志（包含正类概率）
                        log_line = (
                            f"{audio_path}, time:{timestamp:.2f}s, class:{predicted_class}, "
                            f"idx:{pred_idx}, "
                            f"extracted_data:{extracted_array.tolist() if extracted_array is not None else 'None'}, "
                            f"pos_prob:{positive_prob if positive_prob is not None else 'None'}\n"
                        )
                        logf.write(log_line)
                        logf.flush()

    print(f"所有推理结果已保存到 {log_file}")

if __name__ == "__main__":
    # input_dir = "/mnt/f/1.6s_voice/HeyMemo_augment"
    input_dir = "/mnt/f/realtime"
    # input_dir = "/mnt/c/Users/Win11/Downloads/CHiME6"
    run_stream_inference_on_folder(input_dir)