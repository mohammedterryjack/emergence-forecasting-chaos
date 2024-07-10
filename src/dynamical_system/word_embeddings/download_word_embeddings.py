"""
https://github.com/alex-tifrea/poincare_glove?tab=readme-ov-file
https://polybox.ethz.ch/index.php/s/TzX6cXGqCX5KvAn
https://www.kaggle.com/datasets/rtatman/english-word-frequency
"""
from urllib.request import urlretrieve
from pathlib import Path

from pandas import read_csv
from numpy import array

from optimiser import Optimiser
from utils import (
    read_binary_vectors_from_file,
    display_binary_vectors,
    read_dense_vectors_from_file, 
    binarise_vectors_by_threshold, 
    write_vectors_to_file,
    k_most_similar_words
)

def binarise_word_vectors(fname:str,sparse_vector_enlargement_factor:int,n_iterations:int) -> str:
    out_fname = fname.replace('.txt','_binary.txt')
    if not Path(out_fname).exists():
        vocabulary = read_dense_vectors_from_file(
            filename=fname
        )
        dense_word_vectors = array(list(vocabulary.values()))
        vector_size,vector_length,_ = dense_word_vectors.shape
        optimiser = Optimiser(
            vector_length=vector_length, 
            vocabulary_size=vector_size,
            sparse_vector_enlargement_factor=sparse_vector_enlargement_factor,
            n_iterations=n_iterations,
        )
        optimiser.learn_sparse_vectors(dense_vectors=dense_word_vectors)
        optimiser.plot()
        binary_word_vectors = binarise_vectors_by_threshold(vectors=optimiser.sparse_vectors(), threshold=0.0)
        write_vectors_to_file(
            filename=out_fname, 
            vocabulary=list(vocabulary), 
            vectors=binary_word_vectors
        )
    return out_fname

def store_most_frequent_words(
    path_word_frequency_csv:str, 
    path_word_vectors:str,
    n:int
) -> str:
    out_path = path_word_vectors.replace('.txt',f'_filtered_by_{n//1000}k_most_common_words.txt')
    if not Path(out_path).exists():
        data = read_csv(path_word_frequency_csv) 
        common_words = set(data["word"][:n])
        word_vectors = {}
        with open(path_word_vectors) as f:
            for line in f.readlines():
                parts = line.strip().split()
                word = parts[0]
                if word not in common_words:
                    continue
                word_vectors[word] = array(list(map(float,parts[1:])))
        with open(out_path, 'w') as outfile:
            for word, vector in word_vectors.items():
                outfile.write(f"{word} {' '.join(str(round(x,3)) for x in vector)}\n")
    return out_path

def download_file(url:str, out_path:str="") -> str:
    path = Path(out_path)
    path /= Path(url).stem.split('=')[-1]
    fname = f"{path}.txt"
    if not Path(fname).exists():
        urlretrieve(url, fname)
    return fname

vectors_fname = download_file(
    #url="https://polybox.ethz.ch/index.php/s/TzX6cXGqCX5KvAn/download?path=%2F&files=poincare_glove_50x2D_cosh-dist-sq_vocab50k.txt",
    url="https://polybox.ethz.ch/index.php/s/TzX6cXGqCX5KvAn/download?path=%2F&files=vanilla_glove_100D_vocab50k.txt",
    out_path="embeddings"
)
vectors_fname = store_most_frequent_words(
    path_word_frequency_csv="unigram_freq.csv",
    path_word_vectors=vectors_fname,
    n=10000
)
vectors_fname = binarise_word_vectors(
    fname=vectors_fname,
    sparse_vector_enlargement_factor=4,
    n_iterations=20
)
display_binary_vectors(fname=vectors_fname)
results = k_most_similar_words(
    word_vectors=read_binary_vectors_from_file(filename=vectors_fname),
    word="rabbit",
    k=5
)
print(results)