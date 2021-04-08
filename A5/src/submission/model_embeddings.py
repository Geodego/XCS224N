#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XCS224N: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from .cnn import CNN
from .highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab, device, e_char=50):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        @param e_char (int): dimension of the char embedding
        @param device (torch.device)
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### START CODE HERE for part 1f
        self.device = device
        self.vocab = vocab
        self.embed_size = embed_size
        self.char_embed_size = e_char
        pad_token_idx = vocab.char2id['<pad>']
        self.embedding = nn.Embedding(len(vocab.char2id), e_char, padding_idx=pad_token_idx)
        self.cnn = CNN(e_word=self.embed_size, char_dim=self.char_embed_size, device=device)
        self.highway = Highway(e_word=self.embed_size, device=device)
        self.dropout = nn.Dropout(p=0.3)
        ### END CODE HERE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### START CODE HERE for part 1f
        device = input_tensor.device.type
        sentence_length = input_tensor.shape[0]
        batch_size = input_tensor.shape[1]
        max_word_length = input_tensor.shape[2]
        # shape (sentence_length, batch_size, max_word_length, char embed dim)
        sentences_padded = self.embedding(input_tensor)
        # we need to modify sentences_padded to a batch of words with a batch size of sentence_length*batch_size
        x_padded = sentences_padded.reshape((-1, max_word_length, self.char_embed_size))
        # we need to transpose columns to get the required shape for CNN
        # shape (sentence_length*batch_size, char embed dim, max_word_length)
        x_padded = x_padded.transpose(dim0=1, dim1=2)
        x_conv_out = self.cnn(x_padded)
        x_highway = self.highway(x_conv_out)
        output_words = self.dropout(x_highway)
        output_sentences = output_words.reshape((sentence_length, batch_size, self.embed_size))
        return output_sentences
        ### END CODE HERE
