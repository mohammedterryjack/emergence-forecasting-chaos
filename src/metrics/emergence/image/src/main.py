"""docker build -t emergence_metrics . && docker run -p 8080:8080 -it emergence_metrics"""

from enum import Enum 

from numpy import ndarray, roll, zeros
from transfer_entropy import TransferEntropy
from eca import OneDimensionalElementaryCellularAutomata
from matplotlib.pyplot import imshow
from nicegui import ui

ui.slider(min=3, max=300, value=50, step=1, on_change=lambda width:display_ca(width=width.value))

class TransferEntropyNeighbour(Enum):
    LEFT = 1
    RIGHT = -1

def local_transfer_entropy(evolution:ndarray, k_history:int, neighbour:TransferEntropyNeighbour) -> ndarray:   
    metric = TransferEntropy(k=k_history)
    _,width = evolution.shape
    filtered_evolution = zeros(evolution.shape)
    for column_index in range(width):
        filtered_evolution[:,column_index] = metric(
            x=evolution[:,column_index],
            y=roll(evolution,neighbour.value,1)[:,column_index]  
        )
    return filtered_evolution

def display_ca(width:int) -> None:    
    ca =  OneDimensionalElementaryCellularAutomata(
        lattice_width=width,
    )
    for _ in range(500):
        ca.transition(rule_number=110)

    filtered_spacetime = local_transfer_entropy(
        evolution=ca.evolution(), 
        k_history=8, 
        neighbour=TransferEntropyNeighbour.LEFT
    )
    d.clear()
    with d as splitter:
        with splitter.before:
            with ui.pyplot(figsize=(5, 10)):
                imshow(ca.evolution(), cmap='gray')
        with splitter.after:
            with ui.pyplot(figsize=(5, 10)):
                imshow(filtered_spacetime)
            
            
d = ui.splitter()
ui.run()