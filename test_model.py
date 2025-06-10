import os
import torch
import librosa
from model.crnn_model import CnnRnnModel1Channel
from so_mfcc import mfcc

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

def extract_features(wav_path):
    # 加载音频
    audio_data = librosa.load(wav_path, sr=16000)[0]
    # 提取MFCC特征
    mfcc_data = mfcc(y=audio_data, sr=16000, n_mfcc=16, n_mels=40, S=None, norm=None,
                    win_length=512, window='hamming', hop_length=256, n_fft=512,
                    fmin=20, fmax=4050, center=False, power=1, htk=True, dct_type=2, lifter=0,
                    scale_exp=15)
    return torch.FloatTensor(mfcc_data)

def test_folder(model, folder_path, class_names):
    # 统计每类结果
    class_stats = {class_id: {'correct':0, 'total':0} for class_id in class_names}
    
    # 遍历两层文件夹结构
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                # 不区分大小写匹配文件名
                filename_lower = file.lower()  # 使用file变量
                true_class = None  # 初始化true_class
                for class_id, name in class_names.items():
                    if name.lower() in filename_lower:
                        true_class = class_id
                        break
                
                if true_class is not None:
                    try:
                        # 提取特征
                        features = extract_features(file_path)  # 使用file_path而不是filename
                        features = features.unsqueeze(0)  # 添加batch维度
                        
                        # 预测
                        with torch.no_grad():
                            outputs = model(features)
                            _, predicted = torch.max(outputs, 1)
                            predicted_class = predicted.item()
                        
                        # 更新统计
                        class_stats[true_class]['total'] += 1
                        if predicted_class == true_class:
                            class_stats[true_class]['correct'] += 1
                    except Exception as e:
                        print(f"Error processing {file}: {e}")  # 使用file变量
    
    # 打印每类精度
    print(f"{'Class':<15} {'Accuracy':<10} {'Correct/Total'}")
    print("-" * 40)
    for class_id, stats in class_stats.items():
        if stats['total'] > 0:
            acc = 100. * stats['correct'] / stats['total']
            print(f"{class_names[class_id]:<15} {acc:.2f}%     {stats['correct']}/{stats['total']}")

if __name__ == "__main__":
    # 加载模型
    model = load_model('8class_model_best.pth')
    
    # 测试指定文件夹
    test_folder(model, '/mnt/d/project/beskws_tool_release3.4/datas/edgetts_generated/', class_names)
    test_folder(model, '/mnt/d/project/beskws_tool_release3.4/datas/orpheus_generated/', class_names)
    test_folder(model, '/mnt/d/project/beskws_tool_release3.4/datas/human_modified_153/', class_names)