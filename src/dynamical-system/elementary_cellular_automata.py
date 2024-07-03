from collections.abc import Sequence

from numpy import ndarray, roll, stack, apply_along_axis, zeros, binary_repr

class OneDimensionalBinaryCellularAutomata(Sequence):
    def __init__(
        self,
        transition_rule_number:int|None=None,
        neighbourhood_radius:int=1,
        width:int=100,
        time_steps:int=100,
        initial_state:int=0
    ) -> None:
        super().__init__()
        self.transition_rule_number=transition_rule_number
        self.cell_states=2
        self.neighbourhood_radius=neighbourhood_radius
        self.width=width
        self.time_steps=time_steps
        self.initial_state=initial_state
        self.configuration=list(map(int,binary_repr(self.initial_state,self.width)))
        self.evolution = zeros(shape=(self.time_steps, self.width),dtype=int)
    
    def __len__(self):
        return self.time_steps

    def __getitem__(self, i:int) -> ndarray:
        return self.evolution[i]
    
    def set_binary_rule_from_number(self, rule_number:int) -> None:
        local_neighbourhood_size = 2*self.neighbourhood_radius + 1
        binary_string = binary_repr(rule_number, self.cell_states ** local_neighbourhood_size)
        outputs = binary_string[::-1]
        def local_rule(input_neighbourhood:ndarray) -> int:
            lookup_index_binary_str = ''.join(map(str,input_neighbourhood))
            lookup_index = int(lookup_index_binary_str,2)
            return int(outputs[lookup_index])
        self.local_transition_rule = local_rule

    #def set_local_transition_rule(self, rule:callable) -> None:
    #    self.local_transition_rule = rule 
        #TODO: check what rule number would be, set it and return it

    def __repr__(self) -> str:
        return '\n'.join(
            ''.join(
                ("■","□")[cell] for cell in row
            ) for row in self.evolution
        )    
    
    def save(self, fname:str) -> None:
        with open(f"{fname}.txt",'w') as f:
            f.write(str(self))
    
    def evolve(self) -> None:  
        for i in range(self.time_steps):
            self.configuration = self.global_transition(
                configuration=self.configuration
            )
            self.evolution[i, :] = self.configuration

    def global_transition(self, configuration:ndarray) -> ndarray:
        """https://blog.scientific-python.org/matplotlib/elementary-cellular-automata/"""
        local_neighbourhoods = stack(
            [
                roll(configuration,i) 
                for i in range(
                    -self.neighbourhood_radius,
                    self.neighbourhood_radius+1,
                    1
                )
            ]
        )
        return apply_along_axis(self.local_transition_rule, 0, local_neighbourhoods)
    

ca = OneDimensionalBinaryCellularAutomata(
    neighbourhood_radius=1,
    initial_state=1
)
ca.set_binary_rule_from_number(rule_number=110)
ca.evolve()

ca.save('ca')
#print(ca[10])
#print(ca[:30])

from matplotlib.pyplot import imshow, show 
imshow(ca.evolution,cmap='gray')
show()