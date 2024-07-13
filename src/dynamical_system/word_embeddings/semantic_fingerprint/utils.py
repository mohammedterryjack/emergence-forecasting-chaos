from string import punctuation

from numpy import ndarray, array, stack, argsort, argwhere
from matplotlib.pyplot import subplots, show
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import KernelPCA


def normalise_text(text:str) -> str:
    TRANSLATION_TABLE = str.maketrans("","",punctuation)
    return text.lower().translate(TRANSLATION_TABLE) 

def split_paragraphs(text:str) -> list[str]:
    return [
        paragraph for paragraph in map(normalise_text,text.split('\n'))
        if paragraph and not paragraph.isspace()
    ]

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

def k_most_similar_words(word_vectors:dict[str,ndarray], word:str, k:int) -> list[str]:
    vector = word_vectors[word]
    vectors = array(list(word_vectors.values()))
    similarity_scores = vector @ vectors.T
    indexes = argsort(similarity_scores)[-k-1:][::-1]
    print("similarity scores",similarity_scores[indexes])
    return array(list(word_vectors))[indexes]

def words_by_feature_index(feature_index:int, word_vectors:dict[str,ndarray]) -> list[str]:
    vectors = array(list(word_vectors.values()))
    indexes = argwhere(vectors[:,feature_index]).reshape(-1)
    return array(list(word_vectors))[indexes]

def write_binary_vectors_to_file(filename:str, word_vectors:dict[str,ndarray]) -> None:
    with open(filename, 'w') as outfile:
        for word, vector in word_vectors.items():
            outfile.write(f"{word} {' '.join(map(str,vector))}\n")

def read_binary_vectors_from_file(filename:str) -> dict[str,ndarray]:
    vocabulary = {}
    with open(filename) as f:
        for line in f.readlines():
            parts = line.split(' ')
            word = parts[0].strip()
            vector = array(list(map(int, parts[1:])),dtype=int)
            vocabulary[word] = vector
    return vocabulary

def display_binary_vectors(word_vectors:dict[str,ndarray]) -> None:
    vocab = stack(list(word_vectors.values()))
    labels = list(word_vectors)
    n = len(labels)
    _, ax = subplots()
    ax.yaxis.set_major_formatter(lambda tick_val,_: labels[int(tick_val)] if 0 <= int(tick_val)< n else '')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.imshow(vocab)
    show()
