import os
# import sys
# root_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(root_dir)
# sys.path.append('.')
import pdb
import librosa
import torch
import json5
import pickle
import numpy as np
from scipy import io
from quantize_public.quant_crnn_data_flow import crnn_data_flow_quant
from quantize_public.quant_crnn_parameter import quant_model_parameter
from quantize_public.quant_crnn_parameter import verify_quant_param
from quantize_public.batch_gen_quant_data import run_net
if __name__ == "__main__":
    # # print(torch.__version__)
    config_file = "./entercompany_checkpoint/config_8classes_20250815.json5"
    # audio_file = "./entercompany_checkpoint/all_heymemo_short.wav"
    audio_file = "./entercompany_checkpoint/all_heymemo_short.wav"
    dev_pickle = "./entercompany_checkpoint/mixed_quant.pkl"
    checkpoint_path = './entercompany_checkpoint/'
    checkout_file1 = "./entercompany_checkpoint/crnn_model_best_2class1020.pth"

    # 以上为需要具备的输入：
    # config_file :配置文件：*.json5；
    # audio_file :一条音频文件预料：（16k,单声道，.wav,16bit位深）；
    # dev_pickle:随意一条验证集，不参与之前训练的最好,用来验证量化模型的准确率。
    # checkout_file1 :训练出来的模型
    # 
    checkoutpoint1 = torch.load(checkout_file1,weights_only=True)
    checkpoint_name ='crnn_model_best_2class1020.pth'
    checkpoint_quant_name = checkpoint_name.split('.')[0] + '_quant.pt'
    checkpoint_quant_file = os.path.join(checkpoint_path,checkpoint_quant_name)
    param_Q_file = os.path.join(checkpoint_path, checkpoint_name.split('.')[0] + '_quant_params_Q' +'.txt')
    
    with open(config_file, 'r') as f:
        config = json5.load(f)
    # pdb.set_trace()
    # 生成量化模型 checkpoint_quant_name 和param_Q_file这两个文件参与后面运算
    model_quant_thresh = 0.99 #分为0.9~0.999
    float_net, fixed_net = quant_model_parameter(checkoutpoint1, config["feature"], checkpoint_quant_file,param_Q_file,model_quant_thresh)
    
    target_class = config["classes"]
    
    # 对量化模型进行打分查看准确率
    with open(dev_pickle, 'rb') as fid:
        pickle_data = pickle.load(fid)
    
    dev_feature = pickle_data[0].type(torch.float)
    dev_feature = dev_feature.permute(0,2,1)
    dev_label = pickle_data[1]
    idx = (dev_label == 7)
    dev_label[idx] = 0
    idx = (dev_label == 6)
    dev_label[idx] = 0
    idx = (dev_label == 5)
    dev_label[idx] = 0
    idx = (dev_label == 4)
    dev_label[idx] = 0
    idx = (dev_label == 3)
    dev_label[idx] = 0
    idx = (dev_label == 2)
    dev_label[idx] = 0
    verify_quant_param(float_net, fixed_net, dev_feature,dev_label,target_class)
    model_config = config["feature"]
    feature_config = config["feature_input"]
   
    
    #对走入模型的数据流参数进行量化，最终生成kws_weight.c文件
    flow_quant_thresh =0.99 #范围0.9~0.999
    B_list = crnn_data_flow_quant(float_net, pickle_data[0], model_config,flow_quant_thresh)

    signal, _ = librosa.core.load(audio_file, sr=16000)
    signal = 0.1 * signal
    # pdb.set_trace()
    # print(signal.shape)
    dsp_signal = np.zeros(25856,dtype=float)
    # print(dsp_signal.shape)
    frame_num =320
    uframe_num = 256
    frame_cnt = len(signal)/frame_num
    # print("all frame &d" %(int)round(frame_cnt))
    temp_buf=[]
    crnn_out=[]
    crnn_floatout=[]
    feature_data = []
    conv0_out=[]
    conv1_out=[]
    gru_out=[]
    # pdb.set_trace()
    for i in range(round(frame_cnt)):
        # if ((i+1) == 418):
        #     pdb.set_trace()
        used_signal = signal[i*frame_num:((i+1)*frame_num)]
        temp_buf.extend(used_signal)
        if (len(temp_buf)==2*uframe_num):
            signal_input = temp_buf[0:uframe_num*2]
            temp_buf=[]
            dsp_signal=dsp_signal[uframe_num*2:len(dsp_signal)]

        else:
             signal_input = temp_buf[0:uframe_num]
             temp_buf = temp_buf[uframe_num:len(temp_buf)]
        #     pdb.set_trace()
             dsp_signal=dsp_signal[uframe_num:len(dsp_signal)]
        # pdb.set_trace()
        # if i ==100:
            # pdb.set_trace()
        dsp_signal = dsp_signal.tolist() +signal_input
        # pdb.set_trace()
        dsp_signal = np.array(dsp_signal)
        # print(dsp_signal.shape)
        # if (i == 60):
        #     pdb.set_trace()
        # data_in,conv_out,gru_h,fc_out,fc_floatout = run_net(dsp_signal, checkpoint_path, param_Q_file, model_config, feature_config, checkpoint_quant_file, i)
        fc_out = run_net(dsp_signal, checkpoint_path, param_Q_file, model_config, feature_config, checkpoint_quant_file, i)
        # fc_out.extend(prediction)
        # data_int = data_in.transpose(0,2,1)
        # data_in_1 = data_int.flatten() #(1,16,101)维度变为16*101，
        
        # feature_data.extend(data_in_1)
        # conv0=conv_out[0] #（1,16,48）维度
        # # pdb.set_trace()
        # conv0t = conv0.transpose(0,2,1)
        # conv0_1 = conv0t.flatten()
        # conv0_out.extend(conv0_1)
        # conv1 = conv_out[1]
        # conv1t=conv1.transpose(0,2,1)
        # conv1_1 = conv1t.flatten()
        # conv1_out.extend(conv1_1)#长度768
        # gru_out.extend(gru_h)
        # print(gru_h.shape)
        # crnn_out.extend(fc_out)
        # crnn_floatout.extend(fc_floatout)
        # pdb.set_trace()
        # if(i==9):
        #     pdb.set_trace()
    # np.save("./entercompany_checkpoint/feature_data_99.npy",feature_data)
    # pdb.set_trace()
    # io.savemat("./entercompany_checkpoint/feature_quant2.mat",{'array_name':feature_data})
    # io.savemat("./entercompany_checkpoint/conv0_quant2.mat",{'array_name':conv0_out})
    # io.savemat("./entercompany_checkpoint/conv1_quant2.mat",{'array_name':conv1_out})
    # io.savemat("./entercompany_checkpoint/gru_out2.mat",{'array_name':gru_out})
    # # # np.save("./entercompany_checkpoint/gru_out.npy",gru_out)
    # io.savemat("./entercompany_checkpoint/crnn_outquant2.mat",{'array_name':crnn_out})
    # io.savemat("./entercompany_checkpoint/crnn_outfloat2.mat",{'array_name':crnn_floatout})
    # pdb.set_trace()
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
