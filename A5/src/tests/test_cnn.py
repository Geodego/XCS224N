import pytest
import torch
import numpy as np
from A5.src.submission.cnn import CNN

e_word = 2
char_dim = 4
kernel = 5
max_word = 5


@pytest.fixture(scope='class')
def simple_cnn() -> CNN:
    cnn = CNN(e_word=e_word, char_dim=char_dim, kernel=kernel, device='cpu')
    channel1 = torch.ones((1, char_dim, kernel))
    channel2 = torch.ones((1, char_dim, kernel)) * 0.5
    channels = torch.cat((channel1, channel2), dim=0)
    cnn.cnn[0].weight = torch.nn.Parameter(channels)
    cnn.cnn[0].bias = torch.nn.Parameter(torch.zeros(e_word))
    return cnn


def test_cnn(simple_cnn):
    cnn = simple_cnn
    word1 = np.ones((char_dim, max_word))
    word2 = np.ones((char_dim, max_word)) * 2
    words = np.concatenate((np.expand_dims(word1, axis=0), np.expand_dims(word2, axis=0)))

    channel1 = np.ones((char_dim, kernel))
    channel2 = np.ones((char_dim, kernel)) * 0.5

    conv1 = np.array([(word1 * channel1).sum(), (word1 * channel2).sum()]).reshape((1, -1))
    conv2 = np.array([(word2 * channel1).sum(), (word2 * channel2).sum()]).reshape((1, -1))
    expected = np.concatenate((conv1, conv2), axis=0)
    actual = cnn(torch.tensor(words).float())
    assert np.isclose(actual.detach().numpy(), expected).all()


