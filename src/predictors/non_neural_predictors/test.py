from numpy import array, argmax
from matplotlib.pyplot import imshow, show 

from matrix_factorisation import matrix_factorisation_sgd, matrix_factorisation_pseudo_inverse

matrix_mapping_current_id_to_next_id = array(
    [
        [1, 0, 0, 0], 
        [0, 1, 0, 0], 
        [0, 0, 1, 0], 
    ]
)
embeddings_current = array([
    [0,1,1,1,1,0,0,1],
    [1,0,0,1,0,1,1,0],
    [0,0,0,0,1,1,0,1]
])

#embeddings_next = matrix_factorisation_sgd(
embeddings_next = matrix_factorisation_pseudo_inverse(
    sparse_matrix_to_factorise=matrix_mapping_current_id_to_next_id,
    factor_matrix_a=embeddings_current,
)
print(embeddings_next)

imshow(matrix_mapping_current_id_to_next_id)
show() 

imshow(embeddings_current @ embeddings_next)
show()


x_new_embedded = [0,1,0,1,0,1,0,1]
y_new_embedding = x_new_embedded @ embeddings_next
print(y_new_embedding)
y_new = argmax(y_new_embedding)
print(y_new)