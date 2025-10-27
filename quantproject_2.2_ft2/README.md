conv0 out, conv1 out, gru out, fc out, input
[[16, 14], [16, 15], [16, 15], [16, 12], [16, 15]]
fc_out, fc_b_fra 22
[-6684672.  5497856.]
[-1.59375     1.31079102]
new_param_bin number: 21552, 0x5430
bin_all: 21520

.so：最新版，同时生成kws_weight.c，可以对量化后模型进行流式仿真
.so.0：老版本，暂时没用
.so.1：新版本，用于对长音频进行流式仿真，使用时改成.so