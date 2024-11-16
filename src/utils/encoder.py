from enum import Enum
from numpy import ndarray, array, nan_to_num

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
    sequence:list[int], array_size:int, historical_context_length:int
) -> tuple[ndarray,ndarray]:
    emergence_filter = IntegratedInformation(k=historical_context_length) 
    #emergence_filter = TransferEntropy(k=historical_context_length)

    evolution = array([
        eca_encoder(index=index,array_size=array_size)
        for index in sequence
    ])
    if len(sequence) < historical_context_length:
        return evolution[-1], [0. for _ in range(array_size)]

    filtered_evolution = emergence_filter.emergence_filter(evolution=evolution)
    emergent_properties_normalised = nan_to_num(filtered_evolution[-1],nan=0.0)
    max_value = emergent_properties_normalised.max()
    min_value = emergent_properties_normalised.min()
    max_min_range = (max_value - min_value)+1e-9
    emergent_properties_normalised -= min_value
    emergent_properties_normalised /= max_min_range
    return evolution[-1],emergent_properties_normalised


def eca_decoder(lattice:list[float], binary_threshold:float) -> int:
    binary_lattice = (lattice>binary_threshold).astype(int).tolist()
    return ElementaryCellularAutomata.get_state_number_from_binary_lattice(
        binary_lattice=binary_lattice
    )
