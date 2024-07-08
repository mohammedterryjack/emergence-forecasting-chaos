from numpy import stack
from utils import read_binary_vectors_from_file
from matplotlib.pyplot import subplots, show
from matplotlib.ticker import MaxNLocator

vocabulary = read_binary_vectors_from_file(filename='binary_vectors.txt')
vocab = stack(list(vocabulary.values()))
labels = list(vocabulary)
n = len(labels)
fig, ax = subplots()
ax.yaxis.set_major_formatter(lambda tick_val,_: labels[int(tick_val)] if 0 <= int(tick_val)< n else '')
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.imshow(vocab)
show()