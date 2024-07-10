"""https://github.com/mfaruqui/sparse-coding"""

from numpy import array

from optimiser import Optimiser
from utils import (
    read_dense_vectors_from_file, 
    binarise_vectors_by_threshold, 
    write_vectors_to_file
)

vocabulary = read_dense_vectors_from_file(
    filename="dense_vectors.txt"
)
dense_word_vectors = array(list(vocabulary.values()))
vector_size,vector_length,_ = dense_word_vectors.shape
optimiser = Optimiser(
    vector_length=vector_length, 
    vocabulary_size=vector_size,
    sparse_vector_enlargement_factor=10,
    n_iterations=10,
)
optimiser.learn_sparse_vectors(dense_vectors=dense_word_vectors)
optimiser.plot()
binary_word_vectors = binarise_vectors_by_threshold(vectors=optimiser.sparse_vectors(), threshold=0.0)
write_vectors_to_file(
    filename="binary_vectors.txt", 
    vocabulary=list(vocabulary), 
    vectors=binary_word_vectors
)