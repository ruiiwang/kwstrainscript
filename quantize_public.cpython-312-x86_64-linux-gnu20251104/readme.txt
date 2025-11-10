本次量化工程更新主要是为了将量化阈值开出来，方便选择和调试。

model_quant_thresh = 0.99 #分为0.9~0.999
float_net, fixed_net = quant_model_parameter(checkoutpoint1, config["feature"], checkpoint_quant_file,param_Q_file,model_quant_thresh)

 model_quant_thresh量化阈值的含义是，数值越大，说明模型参数的精度损失越大，而超出Q值范围内的数越少。

 flow_quant_thresh =0.99 #范围0.9~0.999
  B_list = crnn_data_flow_quant(float_net, pickle_data[0], model_config,flow_quant_thresh)   
数据流的flow_quant_thresh的阈值含义也是如此。

根据当前模型参数和仿真结果看，之前封装的量化阈值model_quant_thresh默认是0.95，效果不如0.99的结果误差小。因此开参出来供调试使用。
flow_quant_thresh原始为0.99
