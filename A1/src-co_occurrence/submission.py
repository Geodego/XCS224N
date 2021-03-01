#!/usr/bin/env python

import sys
import os
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from utils import *
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(os.path.join('..')))


def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1

    # ### START CODE HERE ###
    corpus_words = sorted(list(set([word for doc in corpus for word in doc])))
    num_corpus_words = len(corpus_words)
    # ### END CODE HERE ###

    return corpus_words, num_corpus_words


def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    
        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.
              
              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".
    
        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)):
                Co-occurrence matrix of word counts. 
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = None
    word2Ind = {}

    # ### START CODE HERE ###
    from collections import Counter
    co_occ = Counter()  # co-occurences counter

    def get_window(pos, doc, window_size):
        """
        return a Counter of each (center_word, word) where word belongs to the window of size window_size, centered
        at center_word. center_word is the word situated in position pos in doc.
        """
        start = max(0, pos - window_size)  # starting cursor of te window
        end = pos + window_size  # ending cursor of the window in doc_ex_center
        end = None if end >= len(doc) - 1 else end

        center_word = doc[pos]
        doc_ex_center = [x for i, x in enumerate(doc) if i != pos]  # remove from doc center_word
        window = doc_ex_center[start:end]
        pairs = [(center_word, x) for x in window]
        return Counter(pairs)

    word2Ind = {word: i for i, word in enumerate(words)}
    M = np.zeros((num_words, num_words))

    for doc in corpus:
        for center_pos in range(len(doc)):
            words_count = get_window(center_pos, doc, window_size)
            co_occ.update(words_count)

    for key, value in co_occ.items():
        M[word2Ind[key[0]], word2Ind[key[1]]] = value
    # ### END CODE HERE ###

    return M, word2Ind


def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurrence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
        Params:
            M (numpy matrix of shape (number of unique words in the corpus , number of number of corpus words)): co-occurrence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """
    np.random.seed(4355)
    n_iters = 10  # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))

    # ### START CODE HERE ###
    from sklearn.decomposition import TruncatedSVD
    M_reduced = TruncatedSVD(k, n_iter=n_iters).fit_transform(M)
    # ### END CODE HERE ###

    print("Done.")
    return M_reduced


def main():
    matplotlib.use('agg')
    plt.rcParams['figure.figsize'] = [10, 5]

    assert sys.version_info[0] == 3
    assert sys.version_info[1] >= 5

    def plot_embeddings(M_reduced, word2Ind, words, title):
        for word in words:
            idx = word2Ind[word]
            x = M_reduced[idx, 0]
            y = M_reduced[idx, 1]
            plt.scatter(x, y, marker='x', color='red')
            plt.text(x, y, word, fontsize=9)
        plt.savefig(title)

    # Read in the corpus
    reuters_corpus = read_corpus()

    M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(reuters_corpus)
    M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)
    # Rescale (normalize) the rows to make them each of unit-length
    M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
    M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis]  # broadcasting

    words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']
    plot_embeddings(M_normalized, word2Ind_co_occurrence, words, 'co_occurrence_embeddings_(soln).png')


if __name__ == "__main__":
    main()
