from urllib.request import urlretrieve
from pathlib import Path

from pandas import read_csv
from numpy import array, ndarray, stack, argsort, argwhere
from matplotlib.pyplot import subplots, show
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import KernelPCA

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

def write_binary_vectors_to_file(filename:str, vocabulary:list[str], vectors:ndarray) -> None:
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

def words_by_feature_index(feature_index:int, word_vectors:dict[str,ndarray]) -> list[str]:
    vectors = array(list(word_vectors.values()))
    indexes = argwhere(vectors[feature_index]).reshape(-1)
    return array(list(word_vectors))[indexes]

def store_most_frequent_words(
    path_word_frequency_csv:str, 
    path_word_vectors:str,
    n_words:int
) -> str:
    out_path = path_word_vectors.replace('.txt',f'_filtered_by_{n_words//1000}k_most_common_words.txt')
    if not Path(out_path).exists():
        data = read_csv(path_word_frequency_csv) 
        common_words = set(data["word"][:n_words])
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

def project_word_vectors_2d(word_vectors:dict[str,ndarray]) -> None:    
    _, ax = subplots()
    coordinates = KernelPCA(
        n_components=2, 
        kernel='cosine'
    ).fit_transform(
        X=array(list(word_vectors.values()))
    )
    for word,(x,y) in zip(word_vectors,coordinates):
        ax.text(x,y,word) 
        ax.plot(x,y)
    show()

def display_spacetime_of_sentence(word_vectors:dict[str,ndarray], sentence:str) -> None:
    labels,vectors = [],[]
    for word in sentence.lower().split():
        if word in word_vectors:
            labels.append(word)
            vectors.append(word_vectors[word])
    n = len(labels)
    _, ax = subplots()
    ax.yaxis.set_major_formatter(lambda tick_val,_: labels[int(tick_val)] if 0 <= int(tick_val)< n else '')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.imshow(stack(vectors))
    show()