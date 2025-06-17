import torch
import pickle
from model.crnn_model import CnnRnnModel1Channel

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
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def test_pkl_data(model, pkl_path, class_names):
    threshold = 0.9  # 可调整的阈值

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    features_list = data[0] # 假设第一个元素是特征列表
    labels_list = data[1] # 假设第二个元素是标签列表

    correct_predictions = 0
    total_predictions = 0

    for i in range(len(features_list)):
        feature = features_list[i].to(torch.float32)  # 将特征数据转换为float32
        true_label = labels_list[i]

        feature = feature.unsqueeze(0)  # 添加batch维度

        with torch.no_grad():
            outputs = model(feature)
            probs = torch.softmax(outputs, 1)
            max_prob, predicted = torch.max(probs, 1)
            predicted_class = predicted.item() if max_prob > threshold else 0
        
        total_predictions += 1
        if predicted_class == true_label:
            correct_predictions += 1
        
        # print(f"Sample {i+1}: True Class: {class_names[int(true_label.item())]}, Predicted Class: {class_names[predicted_class]}, Probability: {max_prob.item():.4f}")

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"Overall Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")

if __name__ == "__main__":
    # 加载模型
    model = load_model('8class_model_best.pth')
    
    # 测试指定pkl文件
    test_pkl_data(model, 'heymemo_devcombine.pkl', class_names)