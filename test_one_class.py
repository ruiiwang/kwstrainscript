import torch
import pickle
from model.crnn_model import CnnRnnModel1Channel
from datetime import datetime

# 模型配置(需与单类别训练时一致)
config = {
    "in_c": 16,
    "conv": [{"out_c": 16, "k": 8, "s": 2, "p": 1, "dropout": 0.0},
            {"out_c": 32, "k": 4, "s": 2, "p": 1, "dropout": 0.0}],
    "rnn": {"dim": 32, "layers": 1, "dropout": 0.2, "bidirectional": True},
    "fc_out": 1  # 更改为1个输出神经元，用于单类别分类
}

# 设备（GPU 优先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 类别名称映射简化为两个
class_names = {
    0: "UNKNOWN_WORD",
    1: "HeyMemo"
}

def _extract_state_dict(ckpt):
    # 从 checkpoint 中提取 state_dict，兼容两种保存格式
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        return ckpt['model_state_dict']
    return ckpt

def _infer_fc_out_from_state_dict(state_dict):
    # 根据 fc.weight 自动推断输出维度
    for k, v in state_dict.items():
        if k.endswith('fc.weight'):
            return v.shape[0]
    # 如果没有找到，默认单输出
    return 1

def load_model(model_path):
    """加载模型并返回（自动推断 fc_out + GPU 支持）"""
    ckpt_cpu = torch.load(model_path, map_location='cpu')
    state_dict_cpu = _extract_state_dict(ckpt_cpu)
    inferred_fc_out = _infer_fc_out_from_state_dict(state_dict_cpu)

    # 使用推断的 fc_out 构建模型
    cfg = dict(config)
    cfg['fc_out'] = inferred_fc_out
    model = CnnRnnModel1Channel(cfg)

    # 按设备加载
    ckpt_dev = torch.load(model_path, map_location=device)
    state_dict_dev = _extract_state_dict(ckpt_dev)
    model.load_state_dict(state_dict_dev)

    model.to(device)
    model.eval()
    return model

def test_pkl_data(model, pkl_path, threshold=0.9, pos_label=1):
    """
    测试pkl文件中的数据，进行单类别唤醒判定。
    始终输出正样本（HeyMemo）的概率：
    - fc_out == 1: sigmoid(logit)
    - fc_out  > 1: softmax(logits)[pos_label]
    """
    print(f"\n--- Testing file: {pkl_path} ---")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    features_list = data[0]
    labels_list = data[1]

    correct_predictions = 0
    total_predictions = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    true_predicted_distribution = {true_name: {pred_name: 0 for pred_name in class_names.values()} for true_name in class_names.values()}

    # 输出维度（用于选择 sigmoid 或 softmax）
    fc_out_dim = int(getattr(model.fc, 'out_features', 1))

    for i in range(len(features_list)):
        feature = features_list[i].to(torch.float32).unsqueeze(0).to(device)  # [1, 16, 100]
        true_lab_raw = labels_list[i].item() if isinstance(labels_list[i], torch.Tensor) else int(labels_list[i])

        # 二值化真实标签（仅关心 HeyMemo vs UNKNOWN）
        true_label_bin = 1 if true_lab_raw == pos_label else 0
        true_name = class_names[true_label_bin]

        with torch.no_grad():
            outputs = model(feature)  # [1, fc_out_dim]

            if fc_out_dim == 1:
                raw_logit = outputs[0, 0].item()
                prob_pos = torch.sigmoid(outputs)[0, 0].item()
            else:
                raw_logit = outputs[0, pos_label].item()
                prob_pos = torch.softmax(outputs, dim=1)[0, pos_label].item()

            predicted_label = 1 if prob_pos > threshold else 0
            predicted_name = class_names[predicted_label]

        total_predictions += 1
        true_predicted_distribution[true_name][predicted_name] += 1

        if predicted_label == true_label_bin:
            correct_predictions += 1

        if true_label_bin == 1 and predicted_label == 1:
            true_positive += 1
        elif true_label_bin == 0 and predicted_label == 1:
            false_positive += 1
        elif true_label_bin == 1 and predicted_label == 0:
            false_negative += 1

        # print(f"Sample {i+1}: True Class: {true_name}, Predicted Class: {predicted_name}, HeyMemo Prob: {prob_pos:.4f}, Raw Logit: {raw_logit:.4f}")

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
    print(f"True Positives: {true_positive}")
    print(f"False Positives: {false_positive}")
    print(f"False Negatives: {false_negative}")

    print("\nTrue Class vs Predicted Class Distribution:")
    header = "{:<20}".format("True Class")
    for pred_name in class_names.values():
        header += "{:<15}".format(pred_name)
    print(header)
    print("-" * len(header))
    for true_name in class_names.values():
        row = "{:<20}".format(true_name)
        for pred_name in class_names.values():
            row += "{:<15}".format(true_predicted_distribution[true_name][pred_name])
        print(row)

if __name__ == "__main__":
    # 加载模型（自动推断 fc_out 并使用 GPU）
    model = load_model('checkpoint_1.0/crnn_model_best.pth')

    # 测试指定pkl文件（阈值可按需调整）
    test_pkl_data(model, 'converted_8/converted_HeyMemo_features.pkl', threshold=0.9)
    test_pkl_data(model, 'converted_un/converted_UNKNOWN_WORD_features.pkl', threshold=0.9)
    test_pkl_data(model, 'mixed_quant.pkl', threshold=0.9)
    test_pkl_data(model, 'heymemo_devcombine.pkl', threshold=0.9)
