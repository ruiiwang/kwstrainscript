import os
# import sys
# root_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(root_dir)
# sys.path.append('.')
import librosa
import torch
import json5
import pickle
from quantize_public.quant_crnn_data_flow import crnn_data_flow_quant
from quantize_public.quant_crnn_parameter import quant_model_parameter
from quantize_public.quant_crnn_parameter import verify_quant_param
from quantize_public.batch_gen_quant_data import run_net
if __name__ == "__main__":
    # print(torch.__version__)
    config_file = "./entercompany_checkpoint/config_2classes_20250909.json5"
    audio_file = "./entercompany_checkpoint/AUS_Sydney_Female_25_HeyMemo_var1.wav"
    dev_pickle = "../mixed_quant.pkl"
    checkpoint_path = './entercompany_checkpoint/'
    checkout_file1 = "./entercompany_checkpoint/crnn_model_best.pth"

    # 以上为需要具备的输入：
    # config_file :配置文件：*.json5；
    # audio_file :一条音频文件预料：（16k,单声道，.wav,16bit位深）；
    # dev_pickle:随意一条验证集，不参与之前训练的最好,用来验证量化模型的准确率。
    # checkout_file1 :训练出来的模型
    # 
    checkoutpoint1 = torch.load(checkout_file1,weights_only=True)
    checkpoint_name ='crnn_model_best.pth'
    checkpoint_quant_name = checkpoint_name.split('.')[0] + '_quant.pt'
    checkpoint_quant_file = os.path.join(checkpoint_path,checkpoint_quant_name)
    param_Q_file = os.path.join(checkpoint_path, checkpoint_name.split('.')[0] + '_quant_params_Q' +'.txt')
    
    with open(config_file, 'r') as f:
        config = json5.load(f)
    
    # 生成量化模型 checkpoint_quant_name 和param_Q_file这两个文件参与后面运算
    float_net, fixed_net = quant_model_parameter(checkoutpoint1, config["feature"], checkpoint_quant_file,param_Q_file)
    
    target_class = config["classes"]
    
    # 对量化模型进行打分查看准确率
    with open(dev_pickle, 'rb') as fid:
        pickle_data = pickle.load(fid)
    
    dev_feature = pickle_data[0].type(torch.float)
    dev_feature = dev_feature.permute(0,2,1)
    dev_label = pickle_data[1]
    # idx = (dev_label == 7)
    # dev_label[idx] = 0
    # idx = (dev_label == 6)
    # dev_label[idx] = 0
    # idx = (dev_label == 5)
    # dev_label[idx] = 0
    # idx = (dev_label == 4)
    # dev_label[idx] = 0
    # idx = (dev_label == 3)
    # dev_label[idx] = 0
    # idx = (dev_label == 2)
    # dev_label[idx] = 0
    verify_quant_param(float_net, fixed_net, dev_feature,dev_label,target_class)
    model_config = config["feature"]
    feature_config = config["feature_input"]
   
    
    #对走入模型的数据流参数进行量化，最终生成kws_weight.c文件
    B_list = crnn_data_flow_quant(float_net, pickle_data[0], model_config)

    signal, _ = librosa.core.load(audio_file, sr=16000)
    # print(len(signal))
    prediction = run_net(signal, checkpoint_path, param_Q_file, model_config, feature_config, checkpoint_quant_file, 0)
    # config_file = "./config/config_wuqi_3target_scaledown_20250901.json5"
    # audio_file = "./entercompany_checkpoint/AUS_Sydney_Female_25_HeyMemo_var1.wav"
    # dev_pickle = "/data/data/datasets_KWS/wuqi_keywords/data_feature/himia_4targeclass_devpicklquant.pkl"
    # checkpoint_path = './checkpoint/2025-09-08_wuqi_3targetclass_scaledown_0813_batch_size4096_mfcc16/'
    # checkout_file1 = "./checkpoint/2025-09-08_wuqi_3targetclass_scaledown_0813_batch_size4096_mfcc16/model_epoch99.pt"

    
    # # 以上为需要具备的输入：
    # # config_file :配置文件：*.json5；
    # # audio_file :一条音频文件预料：（16k,单声道，.wav,16bit位深）；
    # # dev_pickle:随意一条验证集，不参与之前训练的最好,用来验证量化模型的准确率。
    # # checkout_file1 :训练出来的模型
    # # # 
    # checkoutpoint1 = torch.load(checkout_file1,weights_only=True)
    # checkpoint_name ='model_epoch99.pt"'
    # checkpoint_quant_name = checkpoint_name.split('.')[0] + '_quant.pt'
    # checkpoint_quant_file = os.path.join(checkpoint_path,checkpoint_quant_name)
    # param_Q_file = os.path.join(checkpoint_path, checkpoint_name.split('.')[0] + '_quant_params_Q' +'.txt')
    
    # with open(config_file, 'r') as f:
    #     config = json5.load(f)
    
    # # print(checkoutpoint1["model_state_dict"])
    # # 生成量化模型 checkpoint_quant_name 和param_Q_file这两个文件参与后面运算
    # float_net, fixed_net = quant_model_parameter(checkoutpoint1, config["model student"],checkpoint_quant_file,param_Q_file)
    
    # target_class = config["train"]["target_class"]
    
    # # 对量化模型进行打分查看准确率
    # verify_quant_param(float_net, fixed_net, dev_pickle,target_class)
    # model_config = config["model student"]
    # feature_config = config["feature"]
    # with open(dev_pickle, 'rb') as fid:
    #     pickle_data = pickle.load(fid)[0]
    
    
    # #对走入模型的数据流参数进行量化，最终生成kws_weight.c文件
    # B_list = crnn_data_flow_quant(float_net, pickle_data, model_config)
    # prediction = run_net(audio_file, checkpoint_path, param_Q_file, model_config, feature_config, checkpoint_quant_file, 0)
