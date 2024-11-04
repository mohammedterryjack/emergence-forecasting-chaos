from numpy import ndarray, array

from dynamical_system.eca.elementary_cellular_automata import ElementaryCellularAutomata

def eca_encoder(index:int, array_size:int) -> ndarray:
    return array(ElementaryCellularAutomata.create_binary_lattice_from_number(
        state_number=index,
        lattice_width=array_size
    ))

def eca_decoder(lattice:list[float], binary_threshold:float) -> int:
    binary_lattice = (lattice>binary_threshold).astype(int).tolist()
    return ElementaryCellularAutomata.get_state_number_from_binary_lattice(
        binary_lattice=binary_lattice
    )
