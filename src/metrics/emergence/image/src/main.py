"""docker build -t emergence_metrics . && docker run -p 8080:8080 -it emergence_metrics"""

from matplotlib.pyplot import imshow
from nicegui import ui

from eca import OneDimensionalElementaryCellularAutomata
from transfer_entropy import TransferEntropy
from integrated_information import IntegratedInformation
from jpype import startJVM, getDefaultJVMPath

startJVM(getDefaultJVMPath(), f"-Djava.class.path=/app/infodynamics.jar")

def display_ca(width:int, time_steps:int, rule:int, emergence_filter_index:int, k:int) -> None:        
    ca =  OneDimensionalElementaryCellularAutomata(
        lattice_width=width,
    )
    for _ in range(time_steps):
        ca.transition(rule_number=rule)

    emergence_filter = [
        TransferEntropy(k=k).emergence_filter,
        IntegratedInformation(k=k).emergence_filter,
    ][emergence_filter_index]
    filtered_spacetime = emergence_filter(
        evolution=ca.evolution(), 
    )
    
    emergence_window.clear()
    with emergence_window as splitter:
        with splitter.before:
            with ui.pyplot(figsize=(5, 10)):
                imshow(ca.evolution(), cmap='gray')
        with splitter.after:
            with ui.pyplot(figsize=(5, 10)):
                imshow(filtered_spacetime)
            

ui.label('width')
width = ui.slider(
    min=3, 
    max=300, 
    value=50, 
    step=1, 
).props('label-always')
time_steps = ui.number(label='Time Steps', value=500, min=1, max=1000, step=1)
metric = ui.select({0:'Transfer Entropy', 1:'Integrated Information'}, value=0)
k = ui.number(label='K', value=5, min=1, max=7, step=1)
rule = ui.number(label='Rule', value=110, min=0, max=256, step=1)
ui.chip('Run', icon='ads_click', on_click=lambda: display_ca(
    width=width.value,
    time_steps=int(time_steps.value),
    rule=int(rule.value),
    k=int(k.value),
    emergence_filter_index= metric.value,
))
emergence_window = ui.splitter()

ui.run()