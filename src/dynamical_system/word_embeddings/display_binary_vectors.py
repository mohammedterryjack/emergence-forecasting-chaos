from numpy import array, stack

vocabulary = {}
with open('out_vecs.txt') as f:
    for line in f.readlines():
        if line.isspace():
            continue 
        vector = line.split(' ')
        vocabulary[vector[0].strip()] = (array(list(map(float, vector[1:]))) > 0).astype(dtype=int)

from matplotlib.pyplot import subplots, show
from matplotlib.ticker import MaxNLocator

vocab = stack(list(vocabulary.values()))

labels = list(vocabulary)
n = len(labels)
fig, ax = subplots()
ax.yaxis.set_major_formatter(lambda tick_val,_: labels[int(tick_val)] if 0 <= int(tick_val)< n else '')
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.imshow(vocab)
show()