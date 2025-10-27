# analyze_log.py
import re

def analyze_softmax_idx1(log_path):
    values = []
    
    consecutive_count = 0
    current_sequence_pos_probs = []
    consecutive_sequences = []

    with open(log_path, 'r') as f:
        # 在内存中给行列表末尾添加一个空行，以确保文件末尾的连续序列能被正确处理
        lines = f.readlines() + ['']
        # 改为带行号遍历，便于打印具体位置
        for idx, line in enumerate(lines, start=1):
            m_header = re.search(r'^(.*?),\s*time:([-+]?[0-9]*\.?[0-9]+)s', line)
            current_audio = m_header.group(1) if m_header else None
            current_time = float(m_header.group(2)) if m_header else None

            if 'idx:1' in line:
                if consecutive_count == 0:
                    seq_start_time = current_time
                    seq_start_line = idx
                    seq_audio_path = current_audio
                consecutive_count += 1

                m_prob = re.search(r'pos_prob:([-+]?[0-9]*\.?[0-9]+)', line)
                if m_prob:
                    prob_val = float(m_prob.group(1))
                    values.append(prob_val)
                    current_sequence_pos_probs.append(prob_val)

                last_time = current_time
                last_line = idx
            else:
                if consecutive_count > 3:
                    if current_sequence_pos_probs:
                        avg = sum(current_sequence_pos_probs) / len(current_sequence_pos_probs)
                        if avg > 0.9:
                            consecutive_sequences.append([
                                consecutive_count, avg, seq_audio_path, seq_start_line, last_line, seq_start_time, last_time
                            ])
                consecutive_count = 0
                current_sequence_pos_probs = []
                seq_start_time = None
                seq_start_line = None
                last_time = None
                last_line = None
                seq_audio_path = None

    print("--- 连续出现序列 (次数 > 15 平均概率 > 0.9) ---", len(consecutive_sequences))
    for count, avg, audio, start_line, end_line, start_time, end_time in consecutive_sequences:
        print(f"[次数:{count}, 平均值:{avg:.6f}] 文件:{audio} | 行:{start_line}-{end_line} | 时间:{start_time:.2f}s-{end_time:.2f}s")

    if values:
        print("\n--- 总体统计 ---")
        print(f"统计数量: {len(values)}")
        print(f"均值: {sum(values)/len(values):.6f}")
        print(f"最大值: {max(values):.6f}")
        print(f"最小值: {min(values):.6f}")
        filtered_7 = [v for v in values if v > 0.7]
        print(f"pos_prob>0.7的数量: {len(filtered_7)}")
        filtered_9 = [v for v in values if v > 0.9]
        print(f"pos_prob>0.9的数量: {len(filtered_9)}")
    else:
        print("没有找到 idx=1 的数据。")

if __name__ == "__main__":
    analyze_softmax_idx1("voxceleb.log")
