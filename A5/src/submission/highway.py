#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XCS224N: Homework 5
"""

### START CODE HERE for part 1d
import torch
from torch import nn


class Highway(nn.Module):

    def __init__(self, e_word: int):
        """
        Highway skip connection made of:
            A projection layer: linear layer followed by a Relu activation xproj=ReLU(Wprojxconv_out + bproj)
            A gate: linear layer followed by sigmoid activation  xgate = σ(Wgatexconv out + bgate)

        :param e_word: size of final word embedding
        """
        super(Highway, self).__init__()
        self.projection = nn.Sequential(nn.Linear(e_word, e_word), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(e_word, e_word), nn.Sigmoid())
        self.e_word = e_word

    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        """

        :param x_conv_out: batch of word partial embeddings. Tensor of shape(batch_size, e_word)
        :return:
        output of the Highway gate, tensor of shape (batch_size, e_word)
        """
        batch_size = x_conv_out.shape[0]
        x_proj = self.projection(x_conv_out)  # shape (batch_size, e_word)
        x_gate = self.gate(x_conv_out)  # shape (batch_size, e_word)
        x_highway = x_gate * x_proj + (torch.ones((batch_size, self.e_word)) - x_gate) * x_conv_out
        return x_highway

### END CODE HERE
