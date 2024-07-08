from numpy import array, ndarray

def read_binary_vectors_from_file(filename:str) -> dict[str,ndarray]:
    vocabulary = {}
    with open(filename) as f:
        for line in f.readlines():
            parts = line.split(' ')
            word = parts[0].strip()
            vector = array(list(map(int, parts[1:])),dtype=int)
            vocabulary[word] = vector
    return vocabulary

def read_dense_vectors_from_file(filename:str) -> dict[str,ndarray]:
    vocabulary = {}
    with open(filename) as f:
        for line in f.readlines():
            parts = line.strip().split()
            word = parts[0]
            vocabulary[word] = array(list(map(float,parts[1:]))).reshape(-1,1)
    return vocabulary

def write_vectors_to_file(filename:str, vocabulary:list[str], vectors:ndarray) -> None:
    with open(filename, 'w') as outfile:
        for word, vector in zip(vocabulary,vectors):
            outfile.write(f"{word} {' '.join(map(str,vector))}\n")

def binarise_vectors_by_threshold(vectors:ndarray, threshold:float) -> ndarray:
    return (vectors > threshold).astype(dtype=int)
