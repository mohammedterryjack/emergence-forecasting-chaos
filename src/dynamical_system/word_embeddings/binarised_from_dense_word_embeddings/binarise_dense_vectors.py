from pathlib import Path

from numpy import array

from utils import read_dense_vectors_from_file, binarise_vectors_by_threshold, write_vectors_to_file
from optimiser import Optimiser


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