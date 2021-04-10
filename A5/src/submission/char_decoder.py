#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XCS224N: Homework 5
"""

import torch
import torch.nn as nn
from .vocab import VocabEntry


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, device=None, char_embedding_size=50, target_vocab: VocabEntry = None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        @param device (torch.device)
        """
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        ### START CODE HERE for part 2a
        super(CharDecoder, self).__init__()
        if device is None:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.hidden_size = hidden_size
        self.charDecoder = nn.LSTM(input_size=char_embedding_size, hidden_size=hidden_size).to(device)
        target_vocab_size = len(target_vocab.char2id)
        self.char_output_projection = nn.Linear(in_features=hidden_size, out_features=target_vocab_size).to(device)
        pad_token_idx = target_vocab.char2id['<pad>']
        self.pad_token_idx = pad_token_idx
        self.decoderCharEmb = nn.Embedding(target_vocab_size, char_embedding_size,
                                           padding_idx=pad_token_idx).to(device)
        self.target_vocab = target_vocab
        ### END CODE HERE

    def forward(self, input: torch.Tensor, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors
        of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of
        shape (1, batch, hidden_size)
        """
        ### TODO - Implement the forward pass of the character decoder.
        ### START CODE HERE for part 2b
        # dec_hidden initially is (h0, c0) both values set to the combined output vector for the current timestep of
        # the main word-level NMT decoder. Both have shape (1, batch, hidden_size), first dimension is 1
        # as only one state for 1 char is considered.

        # Get the embeddings of the ipput characters
        input_embed = self.decoderCharEmb(input)  # (max length, batch, char embed dim)

        batch = input.shape[1]
        hiddens, (h_n, c_n) = self.charDecoder(input_embed, dec_hidden)  # hiddens (max length, batch, hidden_dim)
        # h_n and c_n shape (1, batch, hidden_dim)

        # project the LSTM output to the target char vocabulary to get logits s
        s = self.char_output_projection(hiddens)  # (max length, batch, tgt char vocab size)
        dec_hidden = (h_n, c_n)
        return s, dec_hidden
        ### END CODE HERE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need
        not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder.
        A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch,
        for every character in the sequence.
        """
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout
        # (e.g., <START>,m,u,s,i,c,<END>).
        ### START CODE HERE for part 2c
        # remove the <end> token for inputs and the <s> token for targets.
        char_input = char_sequence[:-1, :]  # shape (length-1, batch)
        char_targets = char_sequence[1:, :]  # shape (length-1, batch)
        # get the logits  shape (max length-1, batch, tgt char vocab size)
        logits, _ = self.forward(input=char_input, dec_hidden=dec_hidden)

        # calculate log softmax of logits for  CrossEntropy loss
        p = -nn.functional.log_softmax(logits, dim=2)  # shape (max length-1, batch, tgt char vocab size)
        # need to make sure padding characters do not contribute to the cross-entropy loss. For that purpose we make
        padded = (char_sequence[1:, :] != self.pad_token_idx).float()  # 0 for padded char 1 otherwise

        # Compute log probability of generating true target words. Torch.gather as defined keeps only the probability
        # value corresponding to the target word in the vocabulary. Along the 3rd dimension (target vocabulary size)
        # keeps only the value in the vocabulary corresponding to the index of the target word
        # extend padded along vocab size
        p_target = torch.gather(p, dim=2, index=char_targets.unsqueeze(-1)).squeeze(-1)  # (max length-1, batch)
        # set probabilities of names corresponding to padding char to 0
        p_target_adjusted = p_target * padded
        loss_char = p_target_adjusted.sum()  # shape (batch,) Cross entropy loss
        return loss_char
        ### END CODE HERE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the
        # character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        ### START CODE HERE for part 2d
        batch = initialStates[0].shape[1]
        # set up start char embedding for the batch
        current_char = torch.ones((1, batch), device=device,
                                  dtype=torch.long) * self.target_vocab.start_of_word  # (1, batch)
        output_words = current_char  # shape (1, batch)
        if initialStates is None:
            # if dec_hidden is not given we take a current state of zeros
            zero_state = torch.zeros((1, batch, self.hidden_size))
            initialStates = (zero_state, zero_state)

        dec_state = initialStates  # initial state of the decoder
        for t in range(max_length - 1):
            current_char_embed = self.decoderCharEmb(current_char)  # (1, batch, char embed dim)
            _, dec_state = self.charDecoder(current_char_embed, dec_state)
            # project the LSTM output to the target char vocabulary to get logits s_t
            h_t = dec_state[0]  # (1, batch, hidden_dim)
            h_t = h_t.squeeze(0)  # (1, batch, hidden_dim)
            s_t = self.char_output_projection(h_t)  # (batch, tgt char vocab size)
            # apply softmax to logits to get probabilities p_t. dim=-1 means the last dimension is used
            p_t = nn.functional.softmax(s_t, dim=-1)  # (batch, tgt char vocab size)
            current_char = p_t.argmax(dim=1).reshape((1, batch))
            output_words = torch.cat((output_words, current_char), dim=0)  # (t+2, batch)

        # output_words shape is (max_length, batch). Now we need to get the list of words from it.
        # remove the start character
        output_words = output_words[1:, :]
        words_list = output_words.T.tolist()
        vocab = self.target_vocab
        decoded_words = []
        for word in words_list:
            word_str = ''
            for char in word:
                if char == 2:
                    break
                word_str += vocab.id2char[char]
            decoded_words.append(word_str)
        return decoded_words

        ### END CODE HERE
