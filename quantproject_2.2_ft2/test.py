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

def run_stream_inference_on_folder(input_dir, log_file='./heymemo_short.log'):
    checkpoint_path = './entercompany_checkpoint/'
    config_file = os.path.join(checkpoint_path, "config_2classes_20250909.json5")
    
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

    sr = 16000
    window_length = 1.6
    hop_length = 0.02
    window_samples = int(window_length * sr)
    hop_samples = int(hop_length * sr)
    
    # 缓冲区大小，等于一个窗口的采样点数
    buffer_samples = window_samples

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
                    for i in tqdm(range(0, len(processed_signal) - window_samples + 1, hop_samples)):
                        segment = processed_signal[i:i+window_samples]
                        
                        # 计算原始音频中的对应时间戳
                        timestamp = i / sr
                        
                        captured_output = io.StringIO()
                        original_stdout = sys.stdout
                        
                        try:
                            sys.stdout = captured_output
                            prediction = run_net(segment, checkpoint_path, quant_params_file, model_config, feature_config, quantized_model_file, 0)
                            output_text = captured_output.getvalue()
                        finally:
                            sys.stdout = original_stdout

                        # 解析输出，提取两个括号的内容
                        fixed_net = None
                        float_net = None
                        
                        # 查找所有括号内容
                        bracket_matches = re.findall(r'\[([^\]]+)\]', output_text)
                        
                        if len(bracket_matches) >= 2:
                            # 第一个括号作为 fixed_net，第二个括号作为 float_net
                            fixed_net = bracket_matches[0].strip()
                            float_net = bracket_matches[1].strip()
                            
                            # 为了保持与原有softmax计算的兼容性，从float_net提取数值
                            try:
                                numbers = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', float_net)
                                extracted_array = np.array([float(n) for n in numbers])
                            except:
                                extracted_array = None
                        else:
                            print(f"未找到足够的括号数组: {output_text}")
                            extracted_array = None

                        # 对提取出的数据进行 Softmax
                        softmax_result = None
                        if extracted_array is not None:
                            softmax_result = softmax(extracted_array)
                        
                        # 现有代码: 处理预测结果
                        if isinstance(prediction, torch.Tensor):
                            pred_idx = prediction.item() if prediction.numel() == 1 else prediction.argmax().item()
                        else:
                            pred_idx = prediction if isinstance(prediction, int) else int(prediction)
                        
                        predicted_class = CLASS_NAMES.get(pred_idx, f"未知类别 ({pred_idx})")
                        
                        # 将所有结果写入日志
                        log_line = (
                            f"{audio_path}, time:{timestamp:.2f}s, class:{predicted_class}, "
                            f"idx:{pred_idx}, "
                            f"fixed_net:[{fixed_net}], "
                            f"float_net:[{float_net}], "
                            f"softmax_result:{softmax_result.tolist() if softmax_result is not None else 'None'}\n"
                        )
                        logf.write(log_line)
                        logf.flush()

    print(f"所有推理结果已保存到 {log_file}")

if __name__ == "__main__":
    # input_dir = "/mnt/f/realtime"
    input_dir = "../string_heymemo"
    # input_dir = "/mnt/c/Users/Win11/Downloads/CHiME6"
    run_stream_inference_on_folder(input_dir)