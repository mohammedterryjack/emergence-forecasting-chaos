from json import load
from pathlib import Path

from numpy import zeros

from utils import (
    stopwords,
    split_paragraphs,
    read_binary_vectors_from_file,
    write_binary_vectors_to_file,
    k_most_similar_words,
    words_by_feature_index,
    display_binary_vectors,
    display_spacetime_of_sentence,
    project_word_vectors_2d
)


def create_sdrs(documents:dict[str,str], path_save:str, max_paragraphs_per_topics:int = 3) -> None:
    contexts = [
        paragraph for text in documents.values() 
        for paragraph in split_paragraphs(text)[:max_paragraphs_per_topics]
    ]
    sparse_vector_length = len(contexts)

    word_vectors = dict()
    for index,text in enumerate(contexts):
        for word in text.split():
            if word not in word_vectors:
                word_vectors[word] = zeros(sparse_vector_length,dtype=int)
            word_vectors[word][index] = 1

    word_vectors_filtered = dict()
    for word,vector in word_vectors.items():
        if word not in stopwords and vector.sum() > 1:
            word_vectors_filtered[word] = vector

    write_binary_vectors_to_file(
        filename=path_save,
        word_vectors=word_vectors_filtered
    )

vectors_fname = Path('embeddings/sdr_wordnet_vectors_binary.txt')
if not vectors_fname.exists():
    with open('embeddings/wikipedia_documents.json') as f:
        wikipedia_documents = load(f)
    create_sdrs(path_save=str(vectors_fname),documents=wikipedia_documents)
word_vectors = read_binary_vectors_from_file(filename=vectors_fname)
display_binary_vectors(word_vectors=word_vectors)
project_word_vectors_2d(word_vectors=word_vectors)
word = "biology"
results = k_most_similar_words(
    word_vectors=word_vectors,
    word=word,
    k=5
)
print(f"most similar words to '{word}':{results}")
feature_index = 50
results = words_by_feature_index(
    feature_index=feature_index,
    word_vectors=word_vectors
)
print(f"feature {feature_index}:{results}")
display_spacetime_of_sentence(
    word_vectors=word_vectors,
    sentence="once upon a time there lived a boy and his sister in a village they went to fetch some water"
)
#TODO: use sentences from wikipedia entries, paper abstracts, novels, etc