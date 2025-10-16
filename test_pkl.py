import torch
import pickle
from model.crnn_model import CnnRnnModel1Channel

# 模型配置(需与训练时一致)
# 类别名称映射(根据实际类别修改)
class_names = {
    0: "UNKNOWN_WORD",
    1: "HeyMemo",
    # 2: "LookAnd",
    # 3: "Pause",
    # 4: "Play",
    # 5: "StopRecording",
    # 6: "TakeAPicture",
    # 7: "TakeAVideo"
}

config = {
    "in_c": 16,
    "conv": [{"out_c": 16, "k": 8, "s": 2, "p": 1, "dropout": 0.0},
                {"out_c": 32, "k": 4, "s": 2, "p": 1, "dropout": 0.0}],
    "rnn": {"dim": 32, "layers": 1, "dropout": 0.2, "bidirectional": True},
    "fc_out": len(class_names)
}

# 新增：pkl文件中的标签到模型class_names的映射
pkl_label_to_class_name_map = {
    0: 0,  # pkl中的0映射到class_names中的0 (UNKNOWN_WORD)
    1: 1   # pkl中的1映射到class_names中的1 (HeyMemo)
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

def test_pkl_data(model, pkl_path, class_names):
    threshold = 0.9  # 可调整的阈值

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    features_list = data[0] # 假设第一个元素是特征列表
    labels_list = data[1] # 假设第二个元素是标签列表

    correct_predictions = 0
    total_predictions = 0
    
    # 新增：用于统计每个类别的预测数量
    predicted_class_counts = {name: 0 for name in class_names.values()}
    # 新增：用于统计真实类别与预测类别的分布
    true_predicted_distribution = {true_name: {pred_name: 0 for pred_name in class_names.values()} for true_name in class_names.values()}

    for i in range(len(features_list)):
        feature = features_list[i].to(torch.float32)  # 将特征数据转换为float32
        
        # 应用标签转换
        true_label = labels_list[i].item() if isinstance(labels_list[i], torch.Tensor) else labels_list[i]
        true_label = pkl_label_to_class_name_map.get(true_label, 0) # 如果没有映射，默认为UNKNOWN_WORD
        true_label_tensor = torch.tensor(true_label) # 转换为tensor以便后续比较

        feature = feature.unsqueeze(0)  # 添加batch维度

        with torch.no_grad():
            outputs = model(feature)
            probs = torch.softmax(outputs, 1)
            max_prob, predicted = torch.max(probs, 1)
            predicted_class = predicted.item() if max_prob > threshold else 0
        
        total_predictions += 1
        if predicted_class == true_label_tensor.item():
            correct_predictions += 1
            
        # 统计预测类别数量
        predicted_class_counts[class_names[predicted_class]] += 1
        # 统计真实类别与预测类别的分布
        true_predicted_distribution[class_names[true_label]][class_names[predicted_class]] += 1
        
        # print(f"Sample {i+1}: True Class: {class_names[true_label]}, Predicted Class: {class_names[predicted_class]}, Probability: {max_prob.item():.4f}")

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"Overall Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
    
    # 打印每个类别的预测数量
    print("\nPredicted class counts:")
    for class_name, count in predicted_class_counts.items():
        print(f"{class_name}: {count}")

    # 新增：打印真实类别与预测类别的分布
    print("\nTrue Class vs Predicted Class Distribution:")
    # 打印表头
    header = "{:<20}".format("True Class")
    for pred_name in class_names.values():
        header += "{:<15}".format(pred_name)
    print(header)
    # 打印分隔线
    print("-" * len(header))

    # 打印每一行的分布数据
    for true_name in class_names.values():
        row = "{:<20}".format(true_name)
        for pred_name in class_names.values():
            row += "{:<15}".format(true_predicted_distribution[true_name][pred_name])
        print(row)


if __name__ == "__main__":
    # 加载模型
    model = load_model('checkpoint_2.2/crnn_model_best.pth')
    
    # 测试指定pkl文件
    test_pkl_data(model, 'converted_8/converted_HeyMemo_features.pkl', class_names)
    test_pkl_data(model, 'converted_8/converted_LookAnd_features.pkl', class_names)
    test_pkl_data(model, 'converted_8/converted_Pause_features.pkl', class_names)
    test_pkl_data(model, 'converted_8/converted_Play_features.pkl', class_names)
    test_pkl_data(model, 'converted_8/converted_StopRecording_features.pkl', class_names)
    test_pkl_data(model, 'converted_8/converted_TakeAPicture_features.pkl', class_names)
    test_pkl_data(model, 'converted_8/converted_TakeAVideo_features.pkl', class_names)
    # test_pkl_data(model, 'converted_un/wrong_segments2_data.pkl', class_names)

