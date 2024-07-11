#1) [x] topics = list of synsets from wordnet (from max depth to prevent them being too many) 
#2) arrange in order s.t. those with more words in common are closer together (or those with more common branches)
#3) map each snippet to a sdr vector index
#4) for each word, for each text it appears in, or has any type of relation with, set the corresponding vector index to 1,
#5) save this as the word's binary vector and save to file


from pandas import read_csv
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset


def synset_lineage(meaning:Synset) -> set[str]:
    return set(map(
        lambda parent:parent.name(), 
        meaning.closure(lambda parent:parent.hypernyms())
    ))

def k_most_common_words(k:int, path_word_frequency_csv:str) -> list[str]:
    data = read_csv(path_word_frequency_csv) 
    return data["word"][:k]

#contexts = []
#for meaning in wordnet.all_synsets():
#    if 1 <= len(synset_lineage(meaning=meaning)) <= 1:
#        contexts.append(meaning)
#3292
        
for word in k_most_common_words(k=10, path_word_frequency_csv="../unigram_freq.csv"):
    x = wordnet.synset(word)
    print(x)