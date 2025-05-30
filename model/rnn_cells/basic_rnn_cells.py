import torch.nn.functional as F
import torch.nn as nn
import torch

# Implementation from nn._functions.rnn.py
def BasicLSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None,
                  activation=torch.tanh, lst_layer_norm=None):
    '''
    Parameters of a basic LSTM cell
    :param forget_bias: float, the bias added to the forget gates. The biases are for the forget gate in order to
    reduce the scale of forgetting in the beginning of the training. Default 1.0.
    :param activation: activation function of the inner states. Default: torch.tanh()
    :param layer_norm: bool, whether to use layer normalization.
    :return: 
    '''
    hx, cx = hidden

    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    if lst_layer_norm:
        ingate = lst_layer_norm[0](ingate.contiguous())
        forgetgate = lst_layer_norm[1](forgetgate.contiguous())
        cellgate = lst_layer_norm[2](cellgate.contiguous())
        outgate = lst_layer_norm[3](outgate.contiguous())

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = activation(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * activation(cy)

    return hy, cy

###############################################################
###############################################################

# Implementation from nn._functions.rnn.py
def BasicGRUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None,
                  activation=torch.tanh, lst_layer_norm=None):
    '''
    Parameters of a basic GRU cell
    :param forget_bias: float, the bias added to the forget gates. The biases are for the forget gate in order to
    reduce the scale of forgetting in the beginning of the training. Default 1.0.
    :param activation: activation function of the inner states. Default: torch.tanh()
    :param layer_norm: bool, whether to use layer normalization.
    :return: 
    '''

    hidden_size = hidden.size()[-1]
    h_ri_slice = hidden_size*2
    w_hh_ri = w_hh[:h_ri_slice,:]
    b_hh_ri = b_hh[:h_ri_slice]
    w_hh_n = w_hh[h_ri_slice:,:]
    b_hh_n = b_hh[h_ri_slice:]

    gi = F.linear(input, w_ih, b_ih)
    # gh = F.linear(hidden, w_hh, b_hh)
    gh = F.linear(hidden, w_hh_ri, b_hh_ri)
    i_r, i_i, i_n = gi.chunk(3, 1)
    # h_r, h_i, h_n = gh.chunk(3, 1)
    h_r, h_i = gh.chunk(2, 1)

    resetgate_tmp = i_r + h_r
    inputgate_tmp = i_i + h_i
    if lst_layer_norm:
        resetgate_tmp = lst_layer_norm[0](resetgate_tmp.contiguous())
        inputgate_tmp = lst_layer_norm[1](inputgate_tmp.contiguous())

    resetgate = torch.sigmoid(resetgate_tmp)
    inputgate = torch.sigmoid(inputgate_tmp)
    # newgate = activation(i_n + resetgate * h_n)
    newgate = activation(i_n + F.linear(resetgate * hidden, w_hh_n, b_hh_n))

    hy = newgate + inputgate * (hidden - newgate)

    return hy

###############################################################
###############################################################
