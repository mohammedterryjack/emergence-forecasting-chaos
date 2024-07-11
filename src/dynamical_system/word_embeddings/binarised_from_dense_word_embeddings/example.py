"""https://github.com/mfaruqui/sparse-coding"""

from numpy import array, stack
from matplotlib.pyplot import subplots, show
from matplotlib.ticker import MaxNLocator

from optimiser import Optimiser
from dynamical_system.word_embeddings.binarised_from_dense_word_embeddings.utils import (
    read_dense_vectors_from_file, 
    binarise_vectors_by_threshold, 
    write_vectors_to_file,
    read_binary_vectors_from_file
)

vocabulary = read_dense_vectors_from_file(
    filename="embeddings/dense_vectors.txt"
)
dense_word_vectors = array(list(vocabulary.values()))
vector_size,vector_length,_ = dense_word_vectors.shape
optimiser = Optimiser(
    vector_length=vector_length, 
    vocabulary_size=vector_size,
    sparse_vector_enlargement_factor=2,
    n_iterations=20,
)
optimiser.learn_sparse_vectors(dense_vectors=dense_word_vectors)
optimiser.plot()
binary_word_vectors = binarise_vectors_by_threshold(vectors=optimiser.sparse_vectors(), threshold=0.0)
write_vectors_to_file(
    filename="example/binary_vectors.txt", 
    vocabulary=list(vocabulary), 
    vectors=binary_word_vectors
)
vocabulary = read_binary_vectors_from_file(
    filename='example/binary_vectors.txt'
)
vocab = stack(list(vocabulary.values()))
labels = list(vocabulary)
n = len(labels)
fig, ax = subplots()
ax.yaxis.set_major_formatter(lambda tick_val,_: labels[int(tick_val)] if 0 <= int(tick_val)< n else '')
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.imshow(vocab)
show()