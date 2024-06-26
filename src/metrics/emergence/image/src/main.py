"""docker build -t emergence_metrics . && docker run -p 8080:8080 -it emergence_metrics"""

from enum import Enum 

from numpy import ndarray, roll, zeros
from transfer_entropy import TransferEntropy

class TransferEntropyNeighbour(Enum):
    LEFT = 1
    RIGHT = -1

def local_transfer_entropy(evolution:ndarray, k_history:int, neighbour:TransferEntropyNeighbour) -> ndarray:   
    metric = TransferEntropy(k=k_history)
    _,width = evolution.shape
    filtered_evolution = zeros(evolution.shape, dtype=int)
    for column_index in range(width):
        filtered_evolution[:,column_index] = metric(
            x=evolution[:,column_index],
            y=roll(evolution,neighbour.value,1)[:,column_index]  
        )
    return filtered_evolution

from eca import OneDimensionalElementaryCellularAutomata
ca =  OneDimensionalElementaryCellularAutomata(
    lattice_width=100,
)
for _ in range(200):
    ca.transition(rule_number=110)

spacetime = local_transfer_entropy(evolution=ca.evolution(), k_history=1, neighbour=TransferEntropyNeighbour.LEFT)


from matplotlib.pyplot import imshow
from nicegui import ui

with ui.pyplot(figsize=(10, 15)):
    imshow(ca.evolution(), cmap='gray')

with ui.pyplot(figsize=(10, 15)):
    imshow(spacetime, cmap='gray')

ui.run()