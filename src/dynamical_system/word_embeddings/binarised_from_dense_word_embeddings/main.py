"""
https://github.com/alex-tifrea/poincare_glove?tab=readme-ov-file
https://polybox.ethz.ch/index.php/s/TzX6cXGqCX5KvAn
https://www.kaggle.com/datasets/rtatman/english-word-frequency
"""

from binarise_dense_vectors import binarise_word_vectors
from utils import (
    read_binary_vectors_from_file,
    display_binary_vectors,
    k_most_similar_words,
    words_by_feature_index,
    download_file,
    store_most_frequent_words,
    project_word_vectors_2d,
    display_spacetime_of_sentence
)


vectors_fname = download_file(
    url="https://polybox.ethz.ch/index.php/s/TzX6cXGqCX5KvAn/download?path=%2F&files=vanilla_glove_100D_vocab50k.txt",
    out_path="embeddings"
)
vectors_fname = store_most_frequent_words(
   path_word_frequency_csv="unigram_freq.csv",
   path_word_vectors=vectors_fname,
   n_words=-1
)
vectors_fname = binarise_word_vectors(
    fname=vectors_fname,
    sparse_vector_enlargement_factor=4,
    n_iterations=20
)
display_binary_vectors(fname=vectors_fname)
word_vectors = read_binary_vectors_from_file(filename=vectors_fname)
project_word_vectors_2d(word_vectors=word_vectors)
word = "rabbit"
results = k_most_similar_words(
    word_vectors=word_vectors,
    word=word,
    k=5
)
print(f"most similar words to '{word}':{results}")
feature_index = 100
results = words_by_feature_index(
    feature_index=feature_index,
    word_vectors=word_vectors
)
print(f"feature {feature_index}:{results}")
display_spacetime_of_sentence(
    word_vectors=word_vectors,
    sentence="once upon a time there lived a boy and his sister in a village they went to fetch some water"
)