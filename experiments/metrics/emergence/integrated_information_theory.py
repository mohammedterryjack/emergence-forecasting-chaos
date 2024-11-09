from scipy.stats import wasserstein_distance


#TODO:
# - calculate the unconstrained probability distribution of past input states to a cell in any state
# - calculate the unconstrained probability distribution of the future neighbouring states of a cell in any state
# - given the state of a cell:
#   - calculate the cause repetoire - the constrained probability distribution of the past states of the cell's inputs 
#   - calcualte the cause information - the Earth mover's distance between the uconstrained and unconstrained probability distributions
#   - calculate the effect repetoire - the constrained probability distribution of the future states of the neighbouring cells that connect to this cell
#   - calcualte the effect information - the Earth mover's distance between the uconstrained and unconstrained probability distributions
#   - calculate the cause-effect information - the minimum between the cause information and effect information


#TODO: calculate probabilities properly - below isnt right!
neighbourhood_radius = 1
n_cells_in_neighbourhood = 2*neighbourhood_radius + 1
unconstrained_probability_future = unconstrained_probability_past = [1/n_cells_in_neighbourhood for _ in range(n_cells_in_neighbourhood)]

rule = {
    "000":0,
    "001":1,
    "010":1,
    "011":0,
    "100":0,
    "101":0,
    "110":1,
    "111":0
}

total_ones = sum(rule.values())
total_zeros = len(rule) - total_ones
cause_repetoire = {
    0:[1.0/total_zeros if value==0 else 0.0 for value in rule.values() ],
    1:[1.0/total_ones if value==1 else 0.0 for value in rule.values()]
}

total_ones = len(list(filter(lambda key:key[1]=="1",rule)))
total_zeros = len(rule) - total_ones
effect_repetoire = {
    0:[1.0/total_zeros if key[1]=="0" else 0.0 for key in rule],
    1:[1.0/total_ones if key[1]=="1" else 0.0 for key in rule]
}

def cause_information(cell_state:int) -> float:
    return wasserstein_distance(
        u_values=unconstrained_probability_past, 
        v_values=cause_repetoire[cell_state]
    )

def effect_information(cell_state:int) -> float:
    return wasserstein_distance(
        u_values=unconstrained_probability_future, 
        v_values=effect_repetoire[cell_state]
    )

def cause_effect_information(cell_state:int) -> float:
    ci = cause_information(cell_state=cell_state)
    ei = effect_information(cell_state=cell_state)
    return min(ci, ei)

print(cause_information(0), effect_information(0))
print(cause_information(1), effect_information(1))
