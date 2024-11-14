from enum import Enum
from numpy import ndarray, array

from dynamical_system.eca.elementary_cellular_automata import ElementaryCellularAutomata
from metrics.emergence import TransferEntropy, IntegratedInformation


def eca_encoder(
    index:int, array_size:int
) -> ndarray:
    return array(ElementaryCellularAutomata.create_binary_lattice_from_number(
        state_number=index,
        lattice_width=array_size
    ))

def eca_and_emergence_encoder(
    sequence:list[int], array_size:int, max_historical_context_length:int=7
) -> tuple[ndarray,ndarray]:
    historical_context_length = min(len(sequence),max_historical_context_length)
    emergence_filter = IntegratedInformation(k=historical_context_length) #TransferEntropy(k=historical_context_length)

    evolution = array([
        eca_encoder(index=index,array_size=array_size)
        for index in sequence
    ])
    filtered_evolution = emergence_filter.emergence_filter(evolution=evolution)
    return evolution[-1],filtered_evolution[-1]


def eca_decoder(lattice:list[float], binary_threshold:float) -> int:
    binary_lattice = (lattice>binary_threshold).astype(int).tolist()
    return ElementaryCellularAutomata.get_state_number_from_binary_lattice(
        binary_lattice=binary_lattice
    )
