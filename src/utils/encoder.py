from numpy import ndarray, array

from dynamical_system.eca.elementary_cellular_automata import ElementaryCellularAutomata

def eca_encoder(index:int, array_size:int, include_emergent_features:bool=False) -> ndarray:
    binary_lattice = array(ElementaryCellularAutomata.create_binary_lattice_from_number(
        state_number=index,
        lattice_width=array_size
    ))
    #TODO: get real emergent features
    emergent_features = [0.0 for _ in range(array_size)]    
    return binary_lattice + emergent_features if include_emergent_features else binary_lattice

def eca_decoder(lattice:list[float], binary_threshold:float) -> int:
    binary_lattice = (lattice>binary_threshold).astype(int).tolist()
    return ElementaryCellularAutomata.get_state_number_from_binary_lattice(
        binary_lattice=binary_lattice
    )
