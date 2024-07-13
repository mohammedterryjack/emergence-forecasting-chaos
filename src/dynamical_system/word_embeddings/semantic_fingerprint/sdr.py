from string import punctuation
from pathlib import Path 

from pandas import read_csv
from nltk.wsd import lesk
from numpy import zeros, ndarray, array, stack, argsort, argwhere
from matplotlib.pyplot import subplots, show
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import KernelPCA



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
    indexes = argwhere(vectors[feature_index]).reshape(-1)
    return array(list(word_vectors))[indexes]


def k_most_common_words(k:int, path_word_frequency_csv:str) -> list[str]:
    data = read_csv(path_word_frequency_csv) 
    return data["word"][:k].dropna()

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

def normalise_text(text:str) -> str:
    TRANSLATION_TABLE = "".maketrans(punctuation, " " * len(punctuation))
    return text.lower().translate(TRANSLATION_TABLE).replace('-',' ').replace('_',' ')

def create_wordnet_sdr_vectors(
    path_word_frequency_csv:str,
    path_save:str,
) -> None:
    if not Path(path_save).exists():
        #1) [x] topics = list of synsets from wordnet (from max depth to prevent them being too many) 
        #get N most common words as contexts
        #TODO: use wikipedia pages instead
        most_common_synsets = [
            lesk([],word) for word in k_most_common_words(
                k=300, 
                path_word_frequency_csv=path_word_frequency_csv
            )
        ]
        most_common_synsets = list(filter(lambda synset:synset,most_common_synsets))
        contexts = [
            normalise_text(
                f"{synset.name().split('.')[0]} {synset.definition()} {' '.join(synset.examples())}"
            ).split() for synset in most_common_synsets
        ]
        sparse_vector_length = len(contexts)
        #2) arrange in order s.t. those with more words in common are closer together (or those with more common branches)
        #TODO: shortest route problem - travelling salesman optimisation

        #3) for each word, for each text it appears in, or has any type of relation with, set the corresponding vector index to 1,    
        #TODO: go through each word in all contexts and make word vector foe it
        vocabulary = dict()
        words = k_most_common_words(
            k=-1, 
            path_word_frequency_csv=path_word_frequency_csv
        )
        for word in words:
            vector = zeros(sparse_vector_length,dtype=int)
            indexes = [
                index for index,context in enumerate(contexts)
                if word in context
            ]   
            if any(indexes):  
                vector[indexes] = 1
                vocabulary[word] = vector
        
        write_binary_vectors_to_file(
            filename=path_save,
            word_vectors=vocabulary
        )

vectors_fname = 'embeddings/sdr_wordnet_vectors_binary.txt'
create_wordnet_sdr_vectors(
    path_word_frequency_csv="../unigram_freq.csv",
    path_save=vectors_fname,
)
word_vectors = read_binary_vectors_from_file(filename=vectors_fname)
#project_word_vectors_2d(word_vectors=word_vectors)
word = "tree"
results = k_most_similar_words(
    word_vectors=word_vectors,
    word=word,
    k=5
)
print(f"most similar words to '{word}':{results}")
feature_index = 100
results = words_by_feature_index(
    feature_index=feature_index,
    word_vectors=word_vectors
)
print(f"feature {feature_index}:{results}")
display_spacetime_of_sentence(
    word_vectors=word_vectors,
    sentence="once upon a time there lived a boy and his sister in a village they went to fetch some water"
)