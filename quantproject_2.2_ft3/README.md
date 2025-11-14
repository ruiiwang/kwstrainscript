conv0 out, conv1 out, gru out, fc out, input
[[16, 12], [16, 13], [16, 15], [16, 11], [16, 12]]
fc_out, fc_b_fra 19
new_param_bin number: 21552, 0x5430
bin_all: 21520

quant_public.py：可以直接输出量化后音频结果，不需要test.py进行测试
.so：最新版（1104），将量化阈值开出来，方便选择和调试
.so.0：老版本，暂时没用
.so.1：老版本，用于对长音频进行流式仿真，使用时改成.so
.so.2：新版（1022），同时生成kws_weight.c，可以对量化后模型进行流式仿真

用这个multi_command_data_flow_stat.txt
16  13  
16  13  
16  15  
16  11  
16  12  
fc_out, fc_b_fra 20
[-4305408.  4134400.]
[-4.10595703  3.94287109]
new_param_bin number: 21552, 0x5430
bin_all: 21520