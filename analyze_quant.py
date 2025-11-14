import os
import re
import numpy as np

LOG_PATH = "d:/kwstrainscript/test_results/heymemo_0.999_0.999/merged_heymemo.log"
THRESHOLD = 0.9   # 阈值一致率的阈值（只关注编号1）
EPS = 0.1        # 误差容忍范围（只关注编号1）

def parse_merged_log(path):
    times, p1_float, p1_quant = [], [], []
    time_re = re.compile(r'time:(\d+(?:\.\d+)?)')  # 支持无 's' 的写法
    probs_re = re.compile(r'probs:\[(.*?)\]')
    softmax_re = re.compile(r'softmax_result:\[(.*?)\]')
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            t_m = time_re.search(line)
            f_m = probs_re.search(line)
            q_m = softmax_re.search(line)
            if not (t_m and f_m and q_m):
                continue
            try:
                t = float(t_m.group(1))
                f_vals = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?', f_m.group(1))]
                q_vals = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?', q_m.group(1))]
                if len(f_vals) >= 2 and len(q_vals) >= 2:
                    times.append(t)
                    p1_float.append(f_vals[1])  # 只取编号为1的值
                    p1_quant.append(q_vals[1])  # 只取编号为1的值
            except:
                continue
    return np.array(times), np.array(p1_float), np.array(p1_quant)

def compute_metrics(times, p1_f, p1_q, threshold=THRESHOLD, eps=EPS):
    if len(p1_f) == 0 or len(p1_q) == 0:
        return {}

    diff = p1_q - p1_f
    abs_diff = np.abs(diff)
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff**2)))
    bias = float(np.mean(diff))
    within_eps_ratio = float(np.mean(abs_diff <= eps))
    pos_match_ratio = float(np.mean((p1_f >= threshold) == (p1_q >= threshold)))
    try:
        corr = float(np.corrcoef(p1_f, p1_q)[0, 1])
    except Exception:
        corr = float('nan')

    return {
        "frames": int(len(p1_f)),
        "mae": mae,
        "rmse": rmse,
        "bias_mean": bias,
        "corr": corr,
        "within_eps_ratio": within_eps_ratio,
        "pos_match_ratio": pos_match_ratio,
        "threshold": threshold,
        "eps": eps,
        "float_pos_ratio": float(np.mean(p1_f >= threshold)),
        "quant_pos_ratio": float(np.mean(p1_q >= threshold)),
        "max_abs_diff": float(np.max(abs_diff)),
        "p95_abs_diff": float(np.percentile(abs_diff, 95)),
    }

def main():
    times, p1_f, p1_q = parse_merged_log(LOG_PATH)
    m = compute_metrics(times, p1_f, p1_q)
    if not m:
        print("未解析到有效的行或概率数据，请检查日志路径和格式。")
        return
    # 写量化精度日志
    log_path = os.path.join(LOG_DIR, "quant_accuracy.log")
    os.makedirs(LOG_DIR, exist_ok=True)
    
    with open(log_path, "w", encoding="utf-8-sig") as f:
        f.write(f"总帧数: {m['frames']}\n")
        f.write(f"MAE: {m['mae']:.6f}\n")
        f.write(f"RMSE: {m['rmse']:.6f}\n")
        f.write(f"Bias(mean quant - float): {m['bias_mean']:.6f}\n")
        f.write(f"Pearson Corr: {m['corr']:.6f}\n")
        f.write(f"误差<= {m['eps']}: {m['within_eps_ratio']*100:.2f}%\n")
        f.write(f"阈值一致率(T={m['threshold']}): {m['pos_match_ratio']*100:.2f}%\n")
        f.write(f"float阳性率: {m['float_pos_ratio']*100:.2f}%, quant阳性率: {m['quant_pos_ratio']*100:.2f}%\n")
        f.write(f"最大|误差|: {m['max_abs_diff']:.6f}, P95|误差|: {m['p95_abs_diff']:.6f}\n")

if __name__ == "__main__":
    main()