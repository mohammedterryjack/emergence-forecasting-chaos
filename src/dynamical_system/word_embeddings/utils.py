from numpy import array, ndarray, stack, argsort
from matplotlib.pyplot import subplots, show
from matplotlib.ticker import MaxNLocator

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

def display_binary_vectors(fname:str) -> None:
    vocabulary = read_binary_vectors_from_file(
        filename=fname
    )
    vocab = stack(list(vocabulary.values()))
    labels = list(vocabulary)
    n = len(labels)
    _, ax = subplots()
    ax.yaxis.set_major_formatter(lambda tick_val,_: labels[int(tick_val)] if 0 <= int(tick_val)< n else '')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.imshow(vocab)
    show()

def k_most_similar_words(word_vectors:dict[str,ndarray], word:str, k:int) -> list[str]:
    vector = word_vectors[word]
    vectors = array(list(word_vectors.values()))
    similarity_scores = vector @ vectors.T
    indexes = argsort(similarity_scores)[-k-1:][::-1][1:]
    print("similarity scores",similarity_scores[indexes])
    return array(list(word_vectors))[indexes]
