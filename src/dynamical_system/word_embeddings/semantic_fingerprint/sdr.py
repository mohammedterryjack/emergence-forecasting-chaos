#1) [x] topics = list of synsets from wordnet (from max depth to prevent them being too many) 
#2) arrange in order s.t. those with more words in common are closer together (or those with more common branches)
#3) map each snippet to a sdr vector index
#4) for each word, for each text it appears in, or has any type of relation with, set the corresponding vector index to 1,
#5) save this as the word's binary vector and save to file

from json import dumps

from pandas import read_csv
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset
from nltk.wsd import lesk
from numpy import zeros, ndarray, stack
from matplotlib.pyplot import subplots, show
from matplotlib.ticker import MaxNLocator


def synset_lineage(meaning:Synset) -> set[str]:
    return set(map(
        lambda parent:parent.name(), 
        meaning.closure(lambda parent:parent.hypernyms())
    ))

def k_most_common_words(k:int, path_word_frequency_csv:str) -> list[str]:
    data = read_csv(path_word_frequency_csv) 
    return data["word"][:k]

def write_binary_vectors_to_file(filename:str, word_vectors:dict[str,ndarray]) -> None:
    with open(filename, 'w') as outfile:
        for word, vector in word_vectors.items():
            outfile.write(f"{word} {' '.join(map(str,vector))}\n")


def display_binary_vectors(word_vectors:dict[str,ndarray]) -> None:
    vocab = stack(list(word_vectors.values()))
    labels = list(word_vectors)
    n = len(labels)
    _, ax = subplots()
    ax.yaxis.set_major_formatter(lambda tick_val,_: labels[int(tick_val)] if 0 <= int(tick_val)< n else '')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.imshow(vocab)
    show()

def create_wordnet_sdr_vectors(
    path_word_frequency_csv:str,
    path_save:str,
    n_words:int=1000,
) -> None:
    contexts = [
        meaning.name() for meaning in wordnet.all_synsets()
        if 1 <= len(synset_lineage(meaning=meaning)) <= 1
    ]
    with open(path_save.replace('.txt','_featurenames.json'),'w') as f:
        f.write(dumps(contexts,indent=1))
    vector_length = len(contexts) #3292

    vocabulary = dict()
    for word in k_most_common_words(k=n_words, path_word_frequency_csv=path_word_frequency_csv):
        vector = zeros(vector_length)
        synset_names,ids = [],[]
        meaning = lesk([],word)
        if meaning:
            synset_names.append(meaning.name())
        parents = synset_lineage(meaning=meaning) if meaning else set()
        synset_names.extend(parents)
        for synset_name in synset_names:
            if synset_name in contexts:
                ids.append(contexts.index(synset_name))
        vector[ids]=1
        vocabulary[word] = vector
    write_binary_vectors_to_file(
        filename=path_save,
        word_vectors=vocabulary
    )
    display_binary_vectors(
        word_vectors=vocabulary
    )

create_wordnet_sdr_vectors(
    path_word_frequency_csv="../unigram_freq.csv",
    path_save='embeddings/sdr_wordnet_vectors_binary.txt',
)