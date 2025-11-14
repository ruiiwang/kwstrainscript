# 顶部：导入与常量
import os
import wave
import re
from typing import List, Tuple

LOG_PATH = r"d:/kwstrainscript/test_results/heymemo_0.999_0.999/merged_heymemo.log"

# 策略参数（全部写在代码里）
WINDOW_MS = 360          # 窗口长度，毫秒
THRESHOLD = 0.9          # 平均概率阈值
TARGET_CLASS_INDEX = 1   # 使用第1号类别的概率（索引1）
READ_ENCODING = "utf-8"  # 日志读取编码
MIN_GAP_MS = 1000        # 新增：两次唤醒的最小间隔（毫秒），防止连续触发
OUTPUT_LOG_PATH = os.path.join(os.path.dirname(LOG_PATH), "wake_strategy.log")
WRITE_ENCODING = "utf-8-sig"  # 输出日志使用带 BOM 的 UTF-8，避免中文乱码
# 新增：开关（True 开启统计并导出，False 关闭）
SAVE_AUDIO = True
# 新增：音频截取相关常量
AUDIO_PATH = "d:/kwstrainscript/string_sample/heymemo/all_heymemo.wav"  # 如不一致请改这里
SEGMENT_OUTPUT_DIR = "d:/kwstrainscript/test_results/heymemo_missed_float_only"
SEGMENT_TARGET_LEN_SEC = 2.0  # 每段总长
float_pattern = re.compile(r"probs:\[(.*?)\]")
softmax_pattern = re.compile(r"softmax_result:\[(.*?)\]")
time_pattern = re.compile(r"time:(\d+(?:\.\d+)?)")

number_pat = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def parse_prob_list(s: str) -> List[float]:
    # 兼容逗号或空格分隔
    nums = number_pat.findall(s)
    return [float(x) for x in nums]

def parse_log(path: str) -> Tuple[List[float], List[float], List[float]]:
    times: List[float] = []
    float_p1: List[float] = []
    quant_p1: List[float] = []

    with open(path, "r", encoding=READ_ENCODING, errors="ignore") as f:
        for line in f:
            t_m = time_pattern.search(line)
            if not t_m:
                continue
            t = float(t_m.group(1))

            f_m = float_pattern.search(line)
            q_m = softmax_pattern.search(line)
            if not (f_m and q_m):
                # 若某行缺失，跳过
                continue

            f_probs = parse_prob_list(f_m.group(1))
            q_probs = parse_prob_list(q_m.group(1))
            if len(f_probs) <= TARGET_CLASS_INDEX or len(q_probs) <= TARGET_CLASS_INDEX:
                continue

            times.append(t)
            float_p1.append(f_probs[TARGET_CLASS_INDEX])
            quant_p1.append(q_probs[TARGET_CLASS_INDEX])

    if not times:
        raise RuntimeError("未能从日志中解析到任何时间点/概率，请检查日志格式。")
    return times, float_p1, quant_p1

def moving_avg_trailing(values: List[float], times: List[float], window_sec: float) -> List[float]:
    # 使用滑动窗口（尾随窗口）：对 [t - window_sec, t] 区间做平均
    n = len(values)
    avgs = [0.0] * n
    # 前缀和帮助 O(1) 求窗口和
    prefix = [0.0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + values[i]

    j = 0
    for i in range(n):
        # 移动左指针，保证窗口长度 <= window_sec
        while times[i] - times[j] > window_sec and j < i:
            j += 1
        cnt = i - j + 1
        avgs[i] = (prefix[i + 1] - prefix[j]) / cnt
    return avgs

def extract_wake_segments(times: List[float], avg_values: List[float], threshold: float, window_sec: float, min_gap_sec: float) -> List[Tuple[float, float]]:
    # 将平均值超过阈值的区段合并为事件；若相邻事件间隔 < min_gap_sec，则合并为一段
    segments: List[Tuple[float, float]] = []
    above_prev = False
    start_time = None

    for i, avg in enumerate(avg_values):
        above = avg > threshold
        if above and not above_prev:
            start_time = max(0.0, times[i] - window_sec)
        elif not above and above_prev:
            end_time = times[i - 1]
            if not segments:
                segments.append((round(start_time, 2), round(end_time, 2)))
            else:
                prev_start, prev_end = segments[-1]
                gap = start_time - prev_end
                if gap < min_gap_sec:
                    segments[-1] = (prev_start, round(end_time, 2))
                else:
                    segments.append((round(start_time, 2), round(end_time, 2)))
            start_time = None
        above_prev = above

    if above_prev and start_time is not None:
        end_time = times[-1]
        if not segments:
            segments.append((round(start_time, 2), round(end_time, 2)))
        else:
            prev_start, prev_end = segments[-1]
            gap = start_time - prev_end
            if gap < min_gap_sec:
                segments[-1] = (prev_start, round(end_time, 2))
            else:
                segments.append((round(start_time, 2), round(end_time, 2)))
    return segments

# 新增：辅助函数
def segments_no_overlap(float_segments, quant_segments):
    # 返回 float 段中与任一 quant 段无交集的段
    result = []
    q_sorted = sorted(quant_segments)
    for fs in float_segments:
        f_start, f_end = fs
        overlapped = False
        for qs in q_sorted:
            q_start, q_end = qs
            if q_end < f_start:
                continue
            if q_start > f_end:
                break
            # 有交集
            if not (q_end < f_start or q_start > f_end):
                overlapped = True
                break
        if not overlapped:
            result.append(fs)
    return result

def get_wav_duration_seconds(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())

def extend_or_crop_to_target(start: float, end: float, target_len: float, audio_len: float) -> tuple[float, float]:
    # 保证最终长度正好为 target_len，居中优先，边界处按需贴边
    seg_len = end - start
    if seg_len >= target_len:
        mid = (start + end) / 2.0
        new_start = max(0.0, mid - target_len / 2.0)
        new_end = min(audio_len, new_start + target_len)
        # 若尾部因贴边导致长度不足，再向前补齐
        if new_end - new_start < target_len:
            new_start = max(0.0, new_end - target_len)
        return round(new_start, 3), round(new_end, 3)
    else:
        pad = (target_len - seg_len) / 2.0
        new_start = max(0.0, start - pad)
        new_end = min(audio_len, end + pad)
        # 边界贴合后若长度仍不足，另一侧补齐
        if new_end - new_start < target_len:
            if new_start == 0.0:
                new_end = min(audio_len, new_start + target_len)
            elif new_end == audio_len:
                new_start = max(0.0, new_end - target_len)
        return round(new_start, 3), round(new_end, 3)

def cut_wav_segment(src_path: str, start_sec: float, end_sec: float, dst_path: str):
    with wave.open(src_path, "rb") as r:
        fr = r.getframerate()
        n_channels = r.getnchannels()
        sampwidth = r.getsampwidth()
        start_frame = int(start_sec * fr)
        end_frame = int(end_sec * fr)
        r.setpos(start_frame)
        frames = r.readframes(max(0, end_frame - start_frame))
    with wave.open(dst_path, "wb") as w:
        w.setnchannels(n_channels)
        w.setsampwidth(sampwidth)
        w.setframerate(fr)
        w.writeframes(frames)

def main():
    times, float_p1, quant_p1 = parse_log(LOG_PATH)

    # 估计帧间隔（取前两帧差值）
    if len(times) >= 2:
        dt = times[1] - times[0]
    else:
        dt = 0.02  # 回退值
    window_sec = WINDOW_MS / 1000.0

    float_avg = moving_avg_trailing(float_p1, times, window_sec)
    quant_avg = moving_avg_trailing(quant_p1, times, window_sec)

    float_segments = extract_wake_segments(times, float_avg, THRESHOLD, window_sec, MIN_GAP_MS / 1000.0)
    quant_segments = extract_wake_segments(times, quant_avg, THRESHOLD, window_sec, MIN_GAP_MS / 1000.0)

    # 构造输出内容到日志
    out_lines = []
    out_lines.append(f"日志: {LOG_PATH}")
    if len(times) >= 2:
        dt = times[1] - times[0]
    else:
        dt = 0.02
    out_lines.append(f"策略: 窗口={WINDOW_MS}ms, 阈值={THRESHOLD}, 最小间隔={MIN_GAP_MS}ms, 类别索引={TARGET_CLASS_INDEX}, 帧间隔≈{dt:.3f}s")
    out_lines.append("")

    out_lines.append(f"[FLOAT] 唤醒次数: {len(float_segments)}")
    for idx, (s, e) in enumerate(float_segments, 1):
        out_lines.append(f"  [{idx:03d}] [{s:.2f}s ~ {e:.2f}s]  时长={(e - s):.2f}s")

    out_lines.append("")
    out_lines.append(f"[QUANT] 唤醒次数: {len(quant_segments)}")
    for idx, (s, e) in enumerate(quant_segments, 1):
        out_lines.append(f"  [{idx:03d}] [{s:.2f}s ~ {e:.2f}s]  时长={(e - s):.2f}s")

    # 统计 float 唤醒但 quant 未唤醒的段，并导出音频（受 SAVE_AUDIO 控制）
    if SAVE_AUDIO:
        out_lines.append("")
        missed_segments = segments_no_overlap(float_segments, quant_segments)
        out_lines.append(f"[MISSED] float 唤醒但 quant 未唤醒 段数: {len(missed_segments)}")

        if os.path.isfile(AUDIO_PATH):
            os.makedirs(SEGMENT_OUTPUT_DIR, exist_ok=True)
            audio_len = get_wav_duration_seconds(AUDIO_PATH)
            summary_lines = []
            summary_lines.append(f"AUDIO: {AUDIO_PATH}")
            summary_lines.append(f"总长: {audio_len:.2f}s")
            summary_lines.append(f"段数: {len(missed_segments)}")
            summary_lines.append("")

            for idx, (s, e) in enumerate(missed_segments, 1):
                ext_s, ext_e = extend_or_crop_to_target(s, e, SEGMENT_TARGET_LEN_SEC, audio_len)
                out_name = f"missed_{idx:03d}_{ext_s:.2f}-{ext_e:.2f}.wav"
                out_path = os.path.join(SEGMENT_OUTPUT_DIR, out_name)
                cut_wav_segment(AUDIO_PATH, ext_s, ext_e, out_path)
                out_lines.append(f"  [MISSED {idx:03d}] 原段[{s:.2f}s~{e:.2f}s] -> 导出[{ext_s:.2f}s~{ext_e:.2f}s] -> {out_name}")
                summary_lines.append(f"[{idx:03d}] {ext_s:.2f}s ~ {ext_e:.2f}s  (原段 {s:.2f}s ~ {e:.2f}s)")

            with open(os.path.join(SEGMENT_OUTPUT_DIR, "missed_summary.txt"), "w", encoding=WRITE_ENCODING, newline="") as sf:
                sf.write("\n".join(summary_lines) + "\n")
            out_lines.append(f"已导出到文件夹: {SEGMENT_OUTPUT_DIR}")
        else:
            out_lines.append(f"音频不存在，已跳过截取：{AUDIO_PATH}（请设置 AUDIO_PATH）")
    else:
        out_lines.append("")
        out_lines.append("[MISSED] 功能关闭（未统计/未导出）")

    with open(OUTPUT_LOG_PATH, "w", encoding=WRITE_ENCODING, newline="") as wf:
        wf.write("\n".join(out_lines) + "\n")
    print(f"已将结果写入: {OUTPUT_LOG_PATH}")

if __name__ == "__main__":
    main()