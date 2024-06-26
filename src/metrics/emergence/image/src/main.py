"""docker build -t emergence_metrics . && docker run -p 8080:8080 -it emergence_metrics"""

from matplotlib.pyplot import imshow
from nicegui import ui

from eca import OneDimensionalElementaryCellularAutomata
from transfer_entropy import pointwise_transfer_entropy, TransferEntropyNeighbour


def display_ca(width:int) -> None:    
    ca =  OneDimensionalElementaryCellularAutomata(
        lattice_width=width,
    )
    for _ in range(500):
        ca.transition(rule_number=110)

    filtered_spacetime = pointwise_transfer_entropy(
        evolution=ca.evolution(), 
        k_history=8, 
        neighbour=TransferEntropyNeighbour.LEFT
    )
    emergence_window.clear()
    with emergence_window as splitter:
        with splitter.before:
            with ui.pyplot(figsize=(5, 10)):
                imshow(ca.evolution(), cmap='gray')
        with splitter.after:
            with ui.pyplot(figsize=(5, 10)):
                imshow(filtered_spacetime)
            
emergence_window = ui.splitter()
ui.slider(min=3, max=300, value=50, step=1, on_change=lambda width:display_ca(width=width.value))
ui.run()