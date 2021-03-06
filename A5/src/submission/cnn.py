#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XCS224N: Homework 5
"""

### START CODE HERE for part 1e
import torch
from torch import nn


class CNN(nn.Module):

    def __init__(self, e_word: int, char_dim: int, device: torch.device, kernel: int = 5):
        """
        Convolution Network used for the production of character-based word embedding.
        :param kernel: dimension of the kernel of the CNN
        :param e_word: size of final word embedding. It is also the number of channels of the CNN.
        :param char_dim: dimension of char embedding
        :param device: device on which tensor should be attached
        """

        super(CNN, self).__init__()
        self.kernel = kernel
        self.e_word = e_word
        self.char_dim = char_dim
        self.cnn = nn.Sequential(nn.Conv1d(in_channels=char_dim, out_channels=e_word, kernel_size=kernel),
                                 nn.ReLU()).to(device=device)

    def forward(self, x_reshaped: torch.Tensor) -> torch.Tensor:
        """

        :param x_reshaped: tensor representing a batch of padded words embedding.
            shape: (batch_size, char embedding dim, max nber of char in a word(m_word))
        :return:
        Tensor of shape (batch_size, e_word)
        """
        x_reshaped = x_reshaped
        # batch_size = x_reshaped.shape[0]
        # m_word = x_reshaped.shape[2]
        x_conv = self.cnn(x_reshaped)  # shape e_word, m_word-k+1

        # apply max pooling on x_conv
        x_conv_out = x_conv.max(dim=2)[0]
        return x_conv_out

### END CODE HERE
