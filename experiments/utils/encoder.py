from enum import Enum
from numpy import ndarray, array

from dynamical_system.eca.elementary_cellular_automata import ElementaryCellularAutomata

class EncoderOption(Enum):
    SPACETIME_ONLY=0
    EMERGENCE_ONLY=1
    SPACETIME_AND_EMERGENCE=2

def eca_encoder(
    index:int, 
    array_size:int, 
    option:EncoderOption,
) -> ndarray:
    binary_lattice = array(ElementaryCellularAutomata.create_binary_lattice_from_number(
        state_number=index,
        lattice_width=array_size
    ))
    if option == EncoderOption.SPACETIME_ONLY:
        return binary_lattice
    #TODO: get real emergent features
    emergent_features = [0.0 for _ in range(array_size)]    
    if option == EncoderOption.EMERGENCE_ONLY:
        return emergent_features
    return binary_lattice + emergent_features

def eca_decoder(lattice:list[float], binary_threshold:float) -> int:
    binary_lattice = (lattice>binary_threshold).astype(int).tolist()
    return ElementaryCellularAutomata.get_state_number_from_binary_lattice(
        binary_lattice=binary_lattice
    )
