# -*- coding: utf-8 -*-
# @Time: 2021/9/10 10:18
# @Author: yuyinghao
# @FileName: lstm_model.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.optim as optim

layer = 2

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_layer_size
        self.output_size = output_size
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=layer
        )
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.h0 = None
        self.c0 = None

    def forward(self, x):
        r_out, (h, c) = self.lstm(x)
        output = self.out(r_out)

        return output

