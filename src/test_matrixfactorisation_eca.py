from typing import Generator 

from matplotlib.pyplot import subplots, show, tight_layout
from numpy import array, ndarray, zeros, ones
from matplotlib.pyplot import imshow, show 
from scipy.sparse import csr_matrix

from predictors.non_neural_predictors.matrix_factorisation import matrix_factorisation_sgd, matrix_factorisation_pseudo_inverse
from dynamical_system.eca.elementary_cellular_automata import ElementaryCellularAutomata

def construct_sparse_correlation_matrix(indexes:list[int], vector_size:int) -> ndarray:
    sparse_matrix = zeros((vector_size, vector_size))
    current_indices = indexes[:-1]
    next_indices = indexes[1:]
    sparse_matrix[current_indices, next_indices] = 1
    return sparse_matrix

def construct_memory_efficient_sparse_correlation_matrix(indexes:list[int]) -> tuple[ndarray, list[int]]:
    """only creates sparse matrix contianing indices visited in the trajectory 
    - not all possible indices in the configuration space
    which means only the configurations seen can ever be predicted
    this is similar to a lookup table using the vectors instead of the indexes
    which means we can combine the emergent features (as vectors) into the lookup
    """
    original_to_mini_index_mapping = list(set(indexes))
    original_to_mini_index_mapping.sort()
    new_indexes = [original_to_mini_index_mapping.index(i) for i in indexes]

    current_indices = new_indexes[:-1]
    next_indices = new_indexes[1:]

    vector_size = len(original_to_mini_index_mapping)
    sparse_matrix = zeros((vector_size, vector_size))
    sparse_matrix[current_indices, next_indices] = 1

    return sparse_matrix, original_to_mini_index_mapping


def eca_encoder(index:int, array_size:int) -> ndarray:
    return array(ElementaryCellularAutomata.create_binary_lattice_from_number(
        state_number=index,
        lattice_width=array_size
    ))

def predict_next(x:ndarray, trained_embeddings:ndarray) -> ndarray:
    y = x @ trained_embeddings
    return y.argmax()

def predict_n(seed_index:int, n:int, index_to_vector:callable, trained_embeddings:ndarray) -> Generator[tuple[int,ndarray],None,None]:
    index = seed_index   
    for _ in range(n):
        vector = index_to_vector(index=index)
        index = predict_next(
            x=vector,
            trained_embeddings=trained_embeddings
        )
        yield index, vector


ca = ElementaryCellularAutomata(
    lattice_width=50, #17
    time_steps=100,
    transition_rule_number=1
)
metadata = ca.info()

#matrix_mapping_current_id_to_next_id = construct_sparse_correlation_matrix(
#    indexes=metadata.lattice_evolution,
#    vector_size=metadata.lattice_configuration_space
#)
matrix_mapping_current_id_to_next_id,new_index_mapping = construct_memory_efficient_sparse_correlation_matrix(
    indexes=metadata.lattice_evolution
)

#imshow(matrix_mapping_current_id_to_next_id)
#show()

#current_vectors = array([
#    eca_encoder(
#        index=index, array_size=metadata.lattice_width
#    ) for index in range(metadata.lattice_configuration_space)
#])

current_vectors = array([
    eca_encoder(
        index=index, array_size=metadata.lattice_width
    ) for index in new_index_mapping
])


#next_vectors = matrix_factorisation_sgd(
next_vectors = matrix_factorisation_pseudo_inverse(
    sparse_matrix_to_factorise=matrix_mapping_current_id_to_next_id,
    factor_matrix_a=current_vectors,
)

#imshow(next_vectors)
#show()

_, predicted_vectors = zip(*predict_n(
    n=metadata.time_steps,
    seed_index=new_index_mapping.index(metadata.lattice_evolution[0]),#metadata.lattice_evolution[0],
    trained_embeddings=next_vectors,
    index_to_vector=lambda index: eca_encoder(
        index=new_index_mapping[index], #=index, 
        array_size=metadata.lattice_width
    )
))

_, axes = subplots(1, 2, figsize=(10, 10))
axes[0].set_title('Expected')
axes[0].imshow(ca.evolution)
axes[1].set_title('Predicted')
axes[1].imshow(predicted_vectors)

tight_layout()
show()

#TODO: plot like with transformers