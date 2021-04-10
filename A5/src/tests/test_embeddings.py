import pytest
import torch
from A5.src.submission.model_embeddings import ModelEmbeddings
from A5.src.submission.vocab import VocabEntry


@pytest.fixture()
def embedding():
    vocab = VocabEntry()
    emb = ModelEmbeddings(embed_size=4, vocab=vocab, device=torch.device('cpu'))
    return emb


def test_model_embedding(embedding):
    word1 = [["Hello"], ["Hello", "you", "again"]]
    batch_size = len(word1)
    embed_size = embedding.embed_size
    max_len = max([len(x) for x in word1])
    # word1_padded shape (max_sentence_length, batch_size, max_word_length)
    word1_padded = embedding.vocab.to_input_tensor_char(word1, device='cpu')
    word1_embedded = embedding(word1_padded)  # shape (sentence_length, batch_size, embed_size)
    expected_shape = torch.Size([max_len, batch_size, embed_size])
    assert word1_embedded.shape == expected_shape

