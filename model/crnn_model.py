# -*- coding: utf-8 -*-
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)
import torch
import numpy as np
import model.rnn_cells.custom_rnn as CustomRnn
# import pdb

class CnnRnnModel1Channel(torch.nn.Module):
    def __init__(self, config, model_status=0):
        super(CnnRnnModel1Channel, self).__init__()
        
        conv_cfg_list = config["conv"]
        convs = []
        in_c = config["in_c"]
        out_c = 0
        for conv_cfg in conv_cfg_list:
            out_c = conv_cfg["out_c"]
            k = conv_cfg["k"]
            s = conv_cfg["s"]
            p = conv_cfg["p"]
            conv = torch.nn.Conv1d(in_c, out_c, k, stride=s, padding=p)
            if (model_status):
                convs.extend([conv, torch.nn.ReLU(inplace=True)])
            else:
                convs.extend([conv, torch.nn.BatchNorm1d(out_c), torch.nn.ReLU(inplace=True)])

            if conv_cfg["dropout"] != 0:
                convs.append(torch.nn.Dropout(p=conv_cfg["dropout"]))
            in_c = out_c
            
        self.conv = torch.nn.Sequential(*convs)
        rnn_cfg = config["rnn"]

        self.rnn = CustomRnn.custom_GRU(input_size=out_c,
                          hidden_size=rnn_cfg["dim"],
                          num_layers=rnn_cfg["layers"],
                          batch_first=True, 
                          dropout=rnn_cfg["dropout"],
                          bidirectional=rnn_cfg["bidirectional"])
        
        self.fc_in = rnn_cfg["dim"] * 2 * rnn_cfg["layers"] if rnn_cfg["bidirectional"] else rnn_cfg["dim"] * rnn_cfg["layers"]
        self.fc = torch.nn.Linear(self.fc_in, config["fc_out"])


    def forward(self, input):
        # input = input.transpose(1, 2) # 移除此行
        conv_out = self.conv(input)
        conv_out = conv_out.transpose(1, 2)
        rnn_out, h = self.rnn(conv_out)
        h = h.transpose(0, 1)
        h = h.reshape((-1, self.fc_in))
        # h = self.drop(h)
        out = self.fc(h)
        return out


if __name__ == "__main__":
    config = {"in_c" : 16,
              "conv": [{"out_c": 32, "k": 16, "s": 2, "p":5, "dropout": 0.0},
                       {"out_c": 64, "k": 8, "s": 2, "p":3, "dropout": 0.0}],
              "rnn": {"dim": 64, "layers": 1, "dropout": 0.25, "bidirectional": True},
              "fc_out" : 2}
    net = CnnRnnModel1Channel(config)
    data = torch.ones((4096, 100, 16))
    out = net(data)
    # print(net)
    # print(out)
    print(out.size())

    for param in net.parameters():
        print(param.size())

    param_size = 0
    for param in net.named_parameters():
        # print(param[0], '\t', param[1].size())
        param_name = param[0]
        param_var = param[1]
        if ('weight' in param_name):
            if ('conv' in param_name):
                type_length = 2
            else:
                type_length = 1
        else:
            if ('fc' in param_name):
                type_length = 4
            else:
                type_length = 2
            
        size_tmp = 1
        param_shape = param_var.size()
        for i in range(len(param_shape)):
            size_tmp = param_shape[i] * size_tmp
        size_tmp = size_tmp * type_length
        # print(size_tmp)

        param_size = param_size + size_tmp

    param_size = param_size / 1024
    print("param_size: %f K" % param_size)
            
            