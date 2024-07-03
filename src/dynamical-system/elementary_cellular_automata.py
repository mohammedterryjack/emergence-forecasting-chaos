from numpy import ndarray, roll, array, stack, apply_along_axis, zeros, binary_repr

# eca.array() //ndarray
# eca_110[20:] //ndarray
# eca_110[10] //array([10101010111...])
# print(eca_110) //string with black and white emojis
# eca.save('bla.txt')

class OneDimensionalBinaryCellularAutomata:
    def __init__(
        self,
        transition_rule_number:int|None=None,
        neighbourhood_radius:int=1,
        width:int=100,
        time_steps:int=100,
        initial_state:int=0
    ) -> None:
        self.transition_rule_number=transition_rule_number
        self.cell_states=2
        self.neighbourhood_radius=neighbourhood_radius
        self.width=width
        self.time_steps=time_steps
        self.initial_state=initial_state
        self.configuration=list(map(int,binary_repr(self.initial_state,self.width)))
        self.evolution = zeros(shape=(self.time_steps, self.width),dtype=int)
    
    def set_binary_rule_from_number(self, rule_number:int) -> None:
        local_neighbourhood_size = 2*self.neighbourhood_radius + 1
        binary_string = binary_repr(rule_number, self.cell_states ** local_neighbourhood_size)
        outputs = binary_string[::-1]
        def local_rule(input_neighbourhood:ndarray) -> int:
            lookup_index_binary_str = ''.join(map(str,input_neighbourhood))
            lookup_index = int(lookup_index_binary_str,2)
            return int(outputs[lookup_index])
        self.local_transition_rule = local_rule

    def set_local_transition_rule(self, rule:callable) -> None:
        self.local_transition_rule = rule 
        #TODO: check what rule number would be, set it and return it

    def __repr__(self) -> str:
        return '\n'.join(
            ''.join(
                ("■","□")[cell] for cell in row
            ) for row in self.evolution
        )    

    def array(self) -> ndarray:
        return array(self.configuration)
    
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

with open('ca.txt','w') as f:
    f.write(str(ca))

from matplotlib.pyplot import imshow, show 
imshow(ca.evolution,cmap='gray')
show()