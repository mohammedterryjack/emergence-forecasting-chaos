#TODO: tidy this up

from copy import deepcopy
from numpy.linalg import norm
from numpy import arccos, ones_like, ndarray, cos
from random import choice
from eca import OneDimensionalElementaryCellularAutomata
from matplotlib.pyplot import plot, show, xlabel, ylabel, quiver, title
from matplotlib.colors import LinearSegmentedColormap

def cosine_similarity(a:ndarray, b:ndarray) -> float:
    result = a @ b.T
    result /= norm(a)*norm(b)
    return result

def angle(x:ndarray, origin:ndarray|None=None) -> float:
    if origin is None:
        origin = ones_like(x)
        origin[::2] -= 1
    cos_theta = cosine_similarity(a=origin,b=x)
    theta = arccos(cos_theta)
    #theta_derivative = -1 / (1-cos_theta)**0.5
    return theta#, theta_derivative

def density_from_angle(theta:float, vector_length:int) -> float:
    density = vector_length**0.5 * cos(theta)
    return round(density,2)

def density_from_configuration(configuration:list[int]) -> float:
    xnorm = sum(x**2 for x in configuration)**0.5
    density = sum(configuration)/xnorm
    return round(density,2)

rule = 3
T = 10
width = 100

["m","tab:brown","c","g","b"]
cmap = LinearSegmentedColormap.from_list("", ["y","tab:orange","r","tab:pink","tab:purple","tab:blue","tab:gray"])

even_indexes = range(0,width,2)
odd_indexes = range(1,width,2)
initial_configuration = [0 for _ in range(width)]
initial_configuration_ = deepcopy(initial_configuration)

ic = ''.join(map(str,initial_configuration_))
ca =  OneDimensionalElementaryCellularAutomata(
    lattice_width=width,
    initial_configuration= ic
)
for _ in range(T):
    ca.transition(rule_number=rule)


for i in even_indexes:
    initial_configuration[i] = 1
    initial_configuration_ = deepcopy(initial_configuration)
    for j in odd_indexes:
        initial_configuration_[j] = 1
        xs = [
            angle(x=row)
            for row in ca.evolution()
        ]
        us = [y-x for x,y in zip(xs[:-1],xs[1:])]
        arrow_colours = [cmap(index/len(us)) for index in range(len(us))]

        quiver(xs[:-2],xs[1:-1],us[:-1],us[1:],color=arrow_colours)

        ic = ''.join(map(str,initial_configuration_))
        ca =  OneDimensionalElementaryCellularAutomata(
            lattice_width=width,
            initial_configuration= ic
        )
        for _ in range(T):
            ca.transition(rule_number=rule)

initial_configuration = [0 for _ in range(width)]
initial_configuration_ = deepcopy(initial_configuration)
max_even = choice(even_indexes)
max_odd = choice(odd_indexes)
for i in range(0,max_even,2):
    initial_configuration[i] = 1
    initial_configuration_ = deepcopy(initial_configuration)
    for j in range(0,max_odd,2):
        initial_configuration_[j] = 1
ic = ''.join(map(str,initial_configuration_))
ca =  OneDimensionalElementaryCellularAutomata(
    lattice_width=width,
    initial_configuration= ic
)
for _ in range(T):
    ca.transition(rule_number=rule)
xs = [
    angle(x=row)
    for row in ca.evolution()
]

title(ic)
plot(xs[:-1],xs[1:],'-->',color='red')
xlabel("theta(t)")
ylabel("theta(t+1)")
show()

#TODO: draw route of single trajectory on top of this
#TODO: draw streamplot

# from numpy import arange, meshgrid, array
# from matplotlib.pyplot import streamplot

# lookup_table = dict()
# for theta,theta_n,dTheta,dTheta_n in zip(angles[:-1],angles[1:],derivatives[:-1],derivatives[1:]):
#     key = f"{round(theta,1)},{round(dTheta,1)}"
#     result = (theta_n - theta, dTheta_n-dTheta)
#     if key not in lookup_table:
#         lookup_table[key] = []
#     lookup_table[key].append(result)
# lookup_table_ = dict()
# for k,values in lookup_table.items():
#     lookup_table_[k] = (
#         sum(a for a,_ in values)/len(values),
#         sum(b for _,b in values)/len(values)
#     )

# x = arange(-2,2,0.1) 
# y = arange(-2,2,0.1) 
# X,Y = meshgrid(x,y) 
# XY = array([list(f"{round(aa,1)},{round(bb,1)}" for aa,bb in zip(a,b)) for a,b in zip(X,Y)])
# U = array([[lookup_table_.get(ab,(0.0,0.0))[0] for ab in coord] for coord in XY])
# V = array([[lookup_table_.get(ab,(0.0,0.0))[1] for ab in coord] for coord in XY])
# streamplot(X,Y,U,V, density=10, linewidth=None, color='#A23BEC') 
# show()