"""https://blog.scientific-python.org/matplotlib/elementary-cellular-automata/"""

from collections.abc import Sequence
from random import randint
from json import dumps

from numpy import ndarray, roll, stack, apply_along_axis, zeros, binary_repr
from matplotlib.pyplot import imshow, show 
from pydantic import BaseModel

class ECAMetadata(BaseModel):
    cell_states:int
    lattice_width:int
    lattice_configuration_space:int
    time_steps:int
    local_transition_rule:int
    local_transition_neighbourhood_radius:int
    lattice_evolution:list[int]

class ElementaryCellularAutomata(Sequence):
    def __init__(
        self,
        neighbourhood_radius:int=1,
        lattice_width:int=100,
        time_steps:int=100,
        initial_state:int|None = None,
        transition_rule_number:int|None=None,
        representation_zero:str="■",
        representation_one:str="□"
    ) -> None:
        super().__init__()

        assert 0<time_steps, "time steps must be a positive integer"
        self.time_steps=time_steps
        local_neighbourhood_size = neighbourhood_radius*2 + 1
        assert lattice_width>=local_neighbourhood_size, f"lattice width ({lattice_width}) is too small for neighbourhood radius ({neighbourhood_radius})"
        min_state,max_state = 0,2**lattice_width
        if initial_state is None:
            initial_state = randint(min_state,max_state-1)
        assert min_state<=initial_state<max_state, f"initial state ({initial_state}) is out of bounds ({min_state},{max_state})"
        
        min_rule_number,max_rule_number = 0,2 ** 2 ** local_neighbourhood_size
        if transition_rule_number is None:
            transition_rule_number = randint(min_rule_number,max_rule_number-1)
        assert min_rule_number<=transition_rule_number<max_rule_number, f"rule number ({transition_rule_number}) is out of bounds ({min_rule_number},{max_rule_number})"
        local_transition_rule = self.create_binary_rule_from_number(
            rule_number=transition_rule_number,
            local_neighbourhood_size=local_neighbourhood_size
        )
        self.evolution = self.create_spacetime_evolution(
            time_steps=self.time_steps,
            initial_state=initial_state,
            lattice_width=lattice_width,
            neighbourhood_radius=neighbourhood_radius,
            local_transition_rule=local_transition_rule      
        )
        self.info = lambda : ECAMetadata(
            cell_states=2,
            lattice_width=lattice_width,
            lattice_configuration_space=max_state,
            time_steps=time_steps,
            local_transition_rule=transition_rule_number,
            local_transition_neighbourhood_radius=neighbourhood_radius,
            lattice_evolution=list(map(
                self.get_state_number_from_binary_lattice,
                self
            ))
        )
        self.repr_zero = representation_zero
        self.repr_one = representation_one

    def __len__(self) -> int:
        return self.time_steps

    def __getitem__(self, i:int) -> ndarray:
        return self.evolution[i]
    
    def __repr__(self) -> str:
        return dumps(self.info().dict(),indent=2) + '\n' + '\n'.join(
            map(
                lambda configuration:self.stringify_configuration(
                    configuration=configuration,
                    representation_zero=self.repr_zero,
                    representation_one=self.repr_one,
                ),
                self
            )
        )    
    
    def save(self, fname:str) -> None:
        with open(f"{fname}.txt",'w') as f:
            f.write(str(self))

    def show(self) -> None:
        imshow(self.evolution,cmap='gray')
        show()    

    @staticmethod
    def apply_local_transition_rule_to_lattice(
        configuration:ndarray, 
        neighbourhood_radius:int, 
        local_transition_rule:callable
    ) -> ndarray:
        local_neighbourhoods = stack(
            [
                roll(configuration,i) 
                for i in range(
                    -neighbourhood_radius,
                    neighbourhood_radius+1,
                    1
                )
            ]
        )
        return apply_along_axis(local_transition_rule, 0, local_neighbourhoods)

    @staticmethod
    def create_binary_lattice_from_number(state_number:int, lattice_width:int) -> list[int]:
        return list(map(int,binary_repr(state_number,lattice_width)))

    @staticmethod
    def get_state_number_from_binary_lattice(binary_lattice:list[int]) -> int:
        return int(''.join(str(value) for value in binary_lattice),2)

    @staticmethod
    def create_binary_rule_from_number(rule_number:int, local_neighbourhood_size:int) -> callable:
        binary_string = binary_repr(rule_number, 2 ** local_neighbourhood_size)
        outputs = list(map(int,binary_string[::-1]))
        def local_transition_rule(input_neighbourhood:ndarray) -> int:
            assert input_neighbourhood.shape==(local_neighbourhood_size,), f"wrong input dimension. expected ({local_neighbourhood_size},) but got {input_neighbourhood.shape}"
            lookup_index = int(''.join(map(str,input_neighbourhood)),2)
            return outputs[lookup_index]
        return local_transition_rule

    @staticmethod
    def create_spacetime_evolution(time_steps:int, lattice_width:int, initial_state:int, neighbourhood_radius:int, local_transition_rule:callable) -> ndarray: 
        evolution = zeros(shape=(time_steps, lattice_width),dtype=int)  
        evolution[0, :]=ElementaryCellularAutomata.create_binary_lattice_from_number(
            state_number=initial_state,
            lattice_width=lattice_width
        )
        for i in range(1,time_steps):
            evolution[i, :] = ElementaryCellularAutomata.apply_local_transition_rule_to_lattice(
                configuration=evolution[i-1, :],
                neighbourhood_radius=neighbourhood_radius,
                local_transition_rule=local_transition_rule
            )
        return evolution

    @staticmethod
    def stringify_configuration(configuration:ndarray, representation_zero:str, representation_one:str) -> str:
        return ''.join((representation_zero, representation_one)[cell] for cell in configuration)
