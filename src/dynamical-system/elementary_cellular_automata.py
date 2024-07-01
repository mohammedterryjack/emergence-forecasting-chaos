from numpy import ndarray, roll, concatenate, array

# eca.array() //ndarray
# eca_110[20:] //ndarray
# eca_110[10] //array([10101010111...])
# print(eca_110) //string with black and white emojis
# eca.save('bla.txt')

class OneDimensionalCellularAutomata:
    def __init__(
        self,
        transition_rule_number:int|None=None,
        cell_states:int=2,
        neighbourhood_radius:int=1,
        width:int=100,
        time_steps:int=100,
        initial_state:int|None=None
    ) -> None:
        self.transition_rule_number=transition_rule_number
        self.cell_states=cell_states
        self.neighbourhood_radius=neighbourhood_radius
        self.width=width
        self.time_steps=time_steps
        self.initial_state=initial_state
        self.configuration:list[int]=[0 for _ in range(self.width)]

    
    def set_local_transition_rule(self, rule:callable) -> int:
        self.local_transition_rule = rule 
        #TODO: check what rule number would be, set it and return it

    def __repr__(self) -> str:
        pass 

    def array(self) -> ndarray:
        return array(self.configuration)
    
    def evolve(self) -> None:        
        self.evolution = [
            self.global_transition()
            for _ in range(self.time_steps)
        ]

    def global_transition(self) -> list[int]:
        wraparound_lattice = concatenate([
            roll(self.configuration,self.neighbourhood_radius),
            self.configuration[-self.neighbourhood_radius:],
            self.configuration[:self.neighbourhood_radius]
        ])
        neighbourhood_size = (2*self.neighbourhood_radius)+1
        local_neigbourhoods = [
            wraparound_lattice[start_index:start_index+neighbourhood_size]
            for start_index in range(self.width)
        ]
        return list(map(self.local_transition_rule,local_neigbourhoods))
    

ca110 = OneDimensionalCellularAutomata(
    neighbourhood_radius=3
)
ca110.set_local_transition_rule(
    rule=sum
)
ca110.evolve()
from matplotlib.pyplot import imshow, show 
imshow(ca110.evolution)
show()