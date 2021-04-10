import pytest
import torch
from torch import nn
from A5.src.submission.char_decoder import CharDecoder
from A5.src.submission.vocab import VocabEntry


@pytest.fixture()
def char_decoder():
    vocab = VocabEntry()
    decoder = CharDecoder(hidden_size=75, target_vocab=vocab, device=torch.device('cpu'))
    return decoder


def test_init(char_decoder: CharDecoder):
    w = torch.tensor([[3, 4, 7, 9], [3, 6, 8, 11]])
    embedding = char_decoder.decoderCharEmb(w)
    batch_size = len(w)
    max_len = max([len(x) for x in w])
    char_dim = char_decoder.decoderCharEmb.embedding_dim
    expected_shape = torch.Size([batch_size, max_len, char_dim])
    assert embedding.shape == expected_shape


class TestTrainForward:

    def test_train_forward_single(self, char_decoder: CharDecoder):
        w = torch.tensor([[1, 4, 2]]).T
        logits, _ = char_decoder.forward(w[:-1, :])
        p = -nn.functional.log_softmax(logits, dim=2).squeeze(1)
        expected_loss = p[0, 4] + p[1, 2]
        calculated_loss = char_decoder.train_forward(w)
        assert (expected_loss == calculated_loss).item()

    def test_train_forward_single_padded(self, char_decoder: CharDecoder):
        w = torch.tensor([[1, 4, 2, 0, 0]]).T
        logits, _ = char_decoder.forward(w[:-1, :])
        p = -nn.functional.log_softmax(logits, dim=2).squeeze(1)
        expected_loss = p[0, 4] + p[1, 2]
        calculated_loss = char_decoder.train_forward(w)
        assert (expected_loss == calculated_loss).item()

    def test_train_forward_batch(self, char_decoder: CharDecoder):
        w = torch.tensor([[1, 4, 2], [1, 6, 2]]).T
        logits, _ = char_decoder.forward(w[:-1, :])
        p = -nn.functional.log_softmax(logits, dim=2)
        expected_loss = p[0, 0, 4] + p[1, 0, 2] + p[0, 1, 6] + p[1, 1, 2]
        calculated_loss = char_decoder.train_forward(w)
        assert (expected_loss == calculated_loss).item()

    def test_train_forward_batch_padded(self, char_decoder: CharDecoder):
        w = torch.tensor([[1, 4, 2, 0, 0], [1, 6, 2, 0, 0]]).T
        logits, _ = char_decoder.forward(w[:-1, :])
        p = -nn.functional.log_softmax(logits, dim=2)
        expected_loss = p[0, 0, 4] + p[1, 0, 2] + p[0, 1, 6] + p[1, 1, 2]
        calculated_loss = char_decoder.train_forward(w)
        assert (expected_loss == calculated_loss).item()


def test_decode_greedy(char_decoder: CharDecoder):
    hidden_size = char_decoder.hidden_size
    batch = 3
    initial_states = (torch.randn((1, batch, hidden_size)), torch.randn((1, batch, hidden_size)))
    decoded_words = char_decoder.decode_greedy(initial_states, torch.device('cpu'))
    assert len(decoded_words) == batch
    for w in decoded_words:
        assert isinstance(w, str)
    pass

