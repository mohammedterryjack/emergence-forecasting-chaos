from numpy.linalg import norm
from numpy import arccos, ones, ndarray

def convert_zeros_to_minus_one(x:ndarray) -> ndarray:
    return 2*x - 1

def cosine_similarity(a:ndarray, b:ndarray) -> float:
    result = a @ b.T
    result /= (norm(a)*norm(b))+1e-9
    return result

def angle(x:ndarray, origin:ndarray) -> float:
    cos_theta = cosine_similarity(a=origin,b=convert_zeros_to_minus_one(x))
    return arccos(cos_theta)

def projector(embedding:ndarray, lattice_width:int) -> float:
    ref_point = ones(shape=(lattice_width))
    return angle(x=embedding,origin=ref_point)
