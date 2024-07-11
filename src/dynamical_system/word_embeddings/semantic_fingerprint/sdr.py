#1) topics = list of synsets from wordnet (from max depth to prevent them being too many) 
#2) arrange in order s.t. those with more words in common are closer together (or those with more common branches)
#3) map each snippet to a sdr vector index
#4) for each word, for each text it appears in, or has any type of relation with, set the corresponding vector index to 1,
#5) save this as the word's binary vector and save to file