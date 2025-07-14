import os
import torch
import librosa
from model.crnn_model import CnnRnnModel1Channel
from mfcc_io import mfcc

# 模型配置(需与训练时一致)
config = {
    "in_c": 16,
    "conv": [
        {"out_c": 32, "k": 16, "s": 2, "p":5, "dropout": 0.0},
        {"out_c": 64, "k": 8, "s": 2, "p":3, "dropout": 0.0}
    ],
    "rnn": {"dim": 64, "layers": 1, "dropout": 0.25, "bidirectional": True},
    "fc_out": 8  # 8个类别
}

# 类别名称映射(根据实际类别修改)
class_names = {
    0: "UNKNOWN_WORD",
    1: "HeyMemo",
    2: "LookAnd",
    3: "Pause",
    4: "Play",
    5: "StopRecording",
    6: "TakeAPicture",
    7: "TakeAVideo"
}

def load_model(model_path):
    model = CnnRnnModel1Channel(config)
    # 修改此处，加载整个检查点字典，然后提取model_state_dict
    try:
        model.load_state_dict(torch.load(model_path))
    except KeyError:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def extract_features(wav_path):
    # 加载音频并进行音量归一化
    audio_data, sr = librosa.load(wav_path, sr=16000)
    # 添加音量归一化
    audio_data = librosa.util.normalize(audio_data, norm=2)
    # 提取MFCC特征
    mfcc_data = mfcc(y=audio_data, sr=16000, n_mfcc=16, n_mels=40, S=None, norm=None,
                    win_length=512, window='hamming', hop_length=256, n_fft=512,
                    fmin=20, fmax=4050, center=False, power=1, htk=True, dct_type=2, lifter=0,
                    scale_exp=15)
    return torch.FloatTensor(mfcc_data)

def test_folder(model, folder_path, class_names):
    threshold = 0.9  # 可调整的阈值

    # 重新初始化一个更细致的统计结构
    detailed_stats = {
        'true_positives': {cid: 0 for cid in class_names}, # 实际是cid，预测也是cid
        'false_positives': {cid: 0 for cid in class_names}, # 实际不是cid，预测是cid
        'false_negatives': {cid: 0 for cid in class_names}, # 实际是cid，预测不是cid
        'total_actual': {cid: 0 for cid in class_names}, # 实际是cid的总次数
        'total_predicted': {cid: 0 for cid in class_names}, # 预测是cid的总次数
        'actual_unknown_count': 0, # 实际是UNKNOWN_WORD的总次数
        'actual_keyword_count': 0, # 实际是唤醒词的总次数
        'predicted_unknown_as_keyword_count': 0, # 实际是UNKNOWN_WORD，但预测为唤醒词的次数 (误唤)
        'predicted_keyword_as_unknown_count': 0 # 实际是唤醒词，但预测为UNKNOWN_WORD的次数 (漏唤)
    }

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                filename_lower = file.lower()
                true_class = None
                for class_id, name in class_names.items():
                    if name.lower() in filename_lower:
                        true_class = class_id
                        break

                if true_class is not None:
                    try:
                        features = extract_features(file_path)
                        features = features.unsqueeze(0)

                        with torch.no_grad():
                            outputs = model(features)
                            probs = torch.softmax(outputs, 1)
                            max_prob, predicted = torch.max(probs, 1)
                            predicted_class = predicted.item() if max_prob > threshold else 0

                        # 更新详细统计
                        detailed_stats['total_actual'][true_class] += 1
                        detailed_stats['total_predicted'][predicted_class] += 1

                        if true_class == 0: # 实际是UNKNOWN_WORD
                            detailed_stats['actual_unknown_count'] += 1
                            if predicted_class != 0: # 预测为唤醒词 (误唤)
                                detailed_stats['predicted_unknown_as_keyword_count'] += 1
                                detailed_stats['false_positives'][predicted_class] += 1 # 对于被预测的唤醒词，这是FP
                            else: # 预测也是UNKNOWN_WORD
                                detailed_stats['true_positives'][0] += 1
                        else: # 实际是唤醒词
                            detailed_stats['actual_keyword_count'] += 1
                            if predicted_class == true_class: # 正确唤醒
                                detailed_stats['true_positives'][true_class] += 1
                            else: # 预测错误
                                detailed_stats['false_negatives'][true_class] += 1 # 对于实际的唤醒词，这是FN
                                if predicted_class == 0: # 预测为UNKNOWN_WORD (漏唤)
                                    detailed_stats['predicted_keyword_as_unknown_count'] += 1
                                else: # 预测为其他唤醒词 (误识别)
                                    detailed_stats['false_positives'][predicted_class] += 1 # 对于被预测的唤醒词，这是FP

                    except Exception as e:
                        print(f"Error processing {file}: {e}")

    # 打印评估指标
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'FAR':<10} {'FRR':<10} {'TP/Actual'}")
    print("-" * 80)

    # 计算总的FAR和FRR
    far_overall = 0.0
    if detailed_stats['actual_unknown_count'] > 0:
        far_overall = 100. * detailed_stats['predicted_unknown_as_keyword_count'] / detailed_stats['actual_unknown_count']

    frr_overall = 0.0
    if detailed_stats['actual_keyword_count'] > 0:
        frr_overall = 100. * detailed_stats['predicted_keyword_as_unknown_count'] / detailed_stats['actual_keyword_count']

    for class_id, name in class_names.items():
        if class_id == 0: # UNKNOWN_WORD
            # 对于UNKNOWN_WORD，我们主要关注它作为背景音时的误唤率
            # Precision和Recall的概念不直接适用于UNKNOWN_WORD作为唤醒词
            # 这里的TP/Actual显示的是正确识别为UNKNOWN_WORD的比例
            tp_unknown = detailed_stats['true_positives'][0]
            total_actual_unknown = detailed_stats['total_actual'][0]
            acc_unknown = 0.0
            if total_actual_unknown > 0:
                acc_unknown = 100. * tp_unknown / total_actual_unknown
            print(f"{name:<15} {'N/A':<10} {'N/A':<10} {far_overall:.2f}%   {frr_overall:.2f}%   {tp_unknown}/{total_actual_unknown}")
        else: # 唤醒词
            tp = detailed_stats['true_positives'][class_id]
            fp = detailed_stats['false_positives'][class_id]
            fn = detailed_stats['false_negatives'][class_id]
            total_actual = detailed_stats['total_actual'][class_id]
            total_predicted = detailed_stats['total_predicted'][class_id]

            precision = 0.0
            if total_predicted > 0:
                precision = 100. * tp / total_predicted

            recall = 0.0
            if total_actual > 0:
                recall = 100. * tp / total_actual

            print(f"{name:<15} {precision:.2f}%   {recall:.2f}%   {'N/A':<10} {'N/A':<10} {tp}/{total_actual}")


if __name__ == "__main__":
    # 加载模型
    model = load_model('checkpoint1/crnn_model_best.pth')
    
    # 测试指定文件夹
    test_folder(model, '/mnt/d/human_modified_153', class_names)
    test_folder(model, '/mnt/d/1.6svoice', class_names)
