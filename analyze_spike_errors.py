import re
import os
import numpy as np

# 配置
LOG_PATH = "d:/kwstrainscript/test_results/heymemo_0.999_0.999/merged_heymemo.log"
LOW_THRESH = 0.50   # L：浮点近似为 0 的阈值
HIGH_THRESH = 0.90  # H：量化高概率的阈值
HOP_SECONDS = 0.02  # 每帧时间
SMOOTH_N = 1        # 对量化概率做简单平滑窗口（设置为 1 则关闭）
TOP_N_EVENTS = 0   # 输出前 N 个最长异常段

def parse_merged_log(path):
    times, p1_float, p1_quant = [], [], []
    time_re = re.compile(r'time:(\d+(?:\.\d+)?)')
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
                    p1_float.append(f_vals[1])
                    p1_quant.append(q_vals[1])
            except:
                continue
    return np.array(times), np.array(p1_float), np.array(p1_quant)

def moving_avg(arr, n):
    if n <= 1:
        return arr
    kernel = np.ones(n, dtype=float) / n
    return np.convolve(arr, kernel, mode='same')

def contiguous_events(mask):
    events = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            events.append((start, i - 1))
            start = None
    if start is not None:
        events.append((start, len(mask) - 1))
    return events

def main():
    times, p1_f, p1_q = parse_merged_log(LOG_PATH)
    if len(times) == 0:
        print("未解析到数据，请检查 LOG_PATH 与日志格式。")
        return

    p1_q_s = moving_avg(p1_q, SMOOTH_N)
    near_zero_mask = (p1_f < LOW_THRESH)
    spike_mask = (p1_q_s >= HIGH_THRESH)
    violation_mask = near_zero_mask & spike_mask

    total_frames = len(times)
    near_zero_frames = int(np.sum(near_zero_mask))
    spike_frames = int(np.sum(violation_mask))
    spike_ratio_in_near_zero = (spike_frames / near_zero_frames) * 100 if near_zero_frames > 0 else 0.0
    total_spike_seconds = spike_frames * HOP_SECONDS

    # 幅度与差值统计（仅在违规帧上）
    if spike_frames > 0:
        q_on_spike = p1_q_s[violation_mask]
        f_on_spike = p1_f[violation_mask]
        diff_on_spike = q_on_spike - f_on_spike
        stats = {
            "q_mean": float(np.mean(q_on_spike)),
            "q_median": float(np.median(q_on_spike)),
            "q_p95": float(np.percentile(q_on_spike, 95)),
            "q_max": float(np.max(q_on_spike)),
            "diff_mean": float(np.mean(diff_on_spike)),
            "diff_p95": float(np.percentile(diff_on_spike, 95)),
            "diff_max": float(np.max(diff_on_spike)),
        }
    else:
        stats = {k: 0.0 for k in ["q_mean","q_median","q_p95","q_max","diff_mean","diff_p95","diff_max"]}

    # 事件级统计
    events = contiguous_events(violation_mask)
    event_durations = [((end - start + 1) * HOP_SECONDS) for start, end in events]
    events_sorted = sorted(zip(events, event_durations), key=lambda x: x[1], reverse=True)
    event_summary = {
        "count": len(events),
        "mean_len": float(np.mean(event_durations)) if event_durations else 0.0,
        "median_len": float(np.median(event_durations)) if event_durations else 0.0,
        "max_len": max(event_durations) if event_durations else 0.0,
        "total_len": float(np.sum(event_durations)) if event_durations else 0.0,
    }

    # 输出
    print(f"总帧数: {total_frames}")
    print(f"浮点近零帧(L<{LOW_THRESH}): {near_zero_frames} ({near_zero_frames/total_frames*100:.2f}%)")
    print(f"近零帧中的量化高分(H>={HIGH_THRESH})违规比例: {spike_ratio_in_near_zero:.2f}%")
    print(f"违规帧总数: {spike_frames}, 总时长: {total_spike_seconds:.2f}s")
    print(f"量化高分在违规帧上的幅度: mean={stats['q_mean']:.4f}, median={stats['q_median']:.4f}, p95={stats['q_p95']:.4f}, max={stats['q_max']:.4f}")
    print(f"与浮点的差值(quant - float)分布: mean={stats['diff_mean']:.4f}, p95={stats['diff_p95']:.4f}, max={stats['diff_max']:.4f}")
    print(f"事件级统计: count={event_summary['count']}, mean_len={event_summary['mean_len']:.2f}s, median_len={event_summary['median_len']:.2f}s, max_len={event_summary['max_len']:.2f}s, total_len={event_summary['total_len']:.2f}s")

    # 打印最长的TopN事件
    for idx, ((start, end), dur) in enumerate(events_sorted[:TOP_N_EVENTS], 1):
        print(f"TOP{idx}: {times[start]:.2f}s - {times[end]:.2f}s ({dur:.2f}s)")

    # 保存日志到同目录
    out_dir = os.path.dirname(LOG_PATH)
    out_path = os.path.join(out_dir, "spike_errors.log")
    with open(out_path, "w", encoding="utf-8-sig") as f:
        f.write(f"总帧数: {total_frames}\n")
        f.write(f"浮点近零帧(L<{LOW_THRESH}): {near_zero_frames} ({near_zero_frames/total_frames*100:.2f}%)\n")
        f.write(f"近零帧中的量化高分(H>={HIGH_THRESH})违规比例: {spike_ratio_in_near_zero:.2f}%\n")
        f.write(f"违规帧总数: {spike_frames}, 总时长: {total_spike_seconds:.2f}s\n")
        f.write(f"量化高分在违规帧上的幅度: mean={stats['q_mean']:.4f}, median={stats['q_median']:.4f}, p95={stats['q_p95']:.4f}, max={stats['q_max']:.4f}\n")
        f.write(f"与浮点的差值(quant - float): mean={stats['diff_mean']:.4f}, p95={stats['diff_p95']:.4f}, max={stats['diff_max']:.4f}\n")
        f.write(f"事件级: count={event_summary['count']}, mean_len={event_summary['mean_len']:.2f}s, median_len={event_summary['median_len']:.2f}s, max_len={event_summary['max_len']:.2f}s, total_len={event_summary['total_len']:.2f}s\n")
        for idx, ((start, end), dur) in enumerate(events_sorted[:TOP_N_EVENTS], 1):
            f.write(f"TOP{idx}: {times[start]:.2f}s - {times[end]:.2f}s ({dur:.2f}s)\n")
    print(f"已保存到: {out_path}")

if __name__ == "__main__":
    main()