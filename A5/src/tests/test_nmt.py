import pytest
import torch
from A5.src.submission.nmt_model import NMT
from A5.src.submission.vocab import Vocab, VocabEntry



@pytest.fixture()
def data():
    """source and target sentences"""
    source = [['como', 'estas']]
    target = [['how', 'are', 'you']]
    return source, target


@pytest.fixture()
def nmt(data):
    src_vocab = VocabEntry()
    tgt_vocab = VocabEntry()

    # make sure the word in our data belongs to our vocabulary
    for sentence in data[0]:
        for word in sentence:
            src_vocab.add(word)
    for sentence in data[1]:
        for word in sentence:
            tgt_vocab.add(word)
    vocab = Vocab(src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    model = NMT(embed_size=50, hidden_size=60, vocab=vocab)
    return model


def test_forward(nmt, data):
    source, target = data
    output = nmt(source=source, target=target)
    print('load last saved model')

