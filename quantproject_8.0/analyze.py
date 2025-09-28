# analyze_log.py
import re

def analyze_softmax_idx1(log_path):
    values = []
    
    consecutive_count = 0
    current_sequence_softmax_values = []
    consecutive_sequences = []

    with open(log_path, 'r') as f:
        # 在内存中给行列表末尾添加一个空行，以确保文件末尾的连续序列能被正确处理
        lines = f.readlines() + ['']
        for line in lines:
            # 查找 idx:1 的行
            if 'idx:1' in line:
                consecutive_count += 1
                # 提取 softmax_result 数组
                match = re.search(r'softmax_result:\[([^\]]+)\]', line)
                if match:
                    arr_str = match.group(1)
                    arr = [float(x.strip()) for x in arr_str.split(',')]
                    if len(arr) > 1:
                        softmax_val = arr[1]
                        values.append(softmax_val)  # 继续收集所有值用于总体统计
                        current_sequence_softmax_values.append(softmax_val)
            else:
                # 当一行不再是 idx:1，表示一个连续序列结束
                if consecutive_count > 15:
                    if current_sequence_softmax_values:
                        avg = sum(current_sequence_softmax_values) / len(current_sequence_softmax_values)
                        if avg > 0.9:
                            consecutive_sequences.append([consecutive_count, avg])

                # 为下一个序列重置计数器和列表
                consecutive_count = 0
                current_sequence_softmax_values = []

    print("--- 连续出现序列 (次数 > 15 平均概率 > 0.9, [次数, 平均值]) ---", len(consecutive_sequences))
    print(consecutive_sequences)

    if values:
        print("\n--- 总体统计 ---")
        print(f"统计数量: {len(values)}")
        print(f"均值: {sum(values)/len(values):.6f}")
        print(f"最大值: {max(values):.6f}")
        print(f"最小值: {min(values):.6f}")
        # print(f"全部值: {values}")
        filtered_5 = [v for v in values if v > 0.5]
        print(f"softmax_result>0.5的数量: {len(filtered_5)}")
        filtered_9 = [v for v in values if v > 0.9]
        print(f"softmax_result>0.9的数量: {len(filtered_9)}")
    else:
        print("没有找到 idx=1 的数据。")

if __name__ == "__main__":
    analyze_softmax_idx1("realtime2.log")
