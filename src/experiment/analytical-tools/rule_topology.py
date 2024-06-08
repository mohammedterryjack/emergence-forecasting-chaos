#TODO: tidy this up

from copy import deepcopy
from numpy.linalg import norm
from numpy import arccos, ones_like, ndarray, cos

def normalise(values:list[float]) -> list[float]:
    minX = min(values)
    rangeX = max(values)-min(values)
    return [(x-minX)/rangeX for x in values]

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

from eca import OneDimensionalElementaryCellularAutomata
from matplotlib.pyplot import imshow, plot, show, xlabel, ylabel, bar, quiver
from matplotlib.colors import LinearSegmentedColormap

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
    ca.transition(rule_number=rule,)
refpoint = ca.evolution()[0]

for i in even_indexes:
    initial_configuration[i] = 1
    initial_configuration_ = deepcopy(initial_configuration)
    for j in odd_indexes:
        initial_configuration_[j] = 1
        xs = [
            angle(x=row)#,origin=refpoint) 
            for row in ca.evolution()
        ]
        us = [y-x for x,y in zip(xs[:-1],xs[1:])]
        us_normalised = normalise(values=us)
        arrow_colours = [cmap((u**2+v**2)/2) for u,v in zip(us_normalised[:-1],us_normalised[1:])]

        quiver(xs[:-2],xs[1:-1],us[:-1],us[1:],color=arrow_colours)

        ic = ''.join(map(str,initial_configuration_))
        ca =  OneDimensionalElementaryCellularAutomata(
            lattice_width=width,
            initial_configuration= ic
        )
        for _ in range(T):
            ca.transition(rule_number=rule)




xlabel("theta(t)")
ylabel("theta(t+1)")
show()



# all_angles, all_derivatives = [],[]
# density_frequency = {}
# for n in range(500):
#     if 0 <= n < 100:
#         i = "1"*n + "0"*(100-n)
#         ca = OneDimensionalElementaryCellularAutomata(lattice_width=100, initial_configuration=i)
#     elif 100 <= n < 200:
#         m = n-100
#         i = "0"*m + "1"*(100-m)
#         ca = OneDimensionalElementaryCellularAutomata(lattice_width=100, initial_configuration=i)
#     else:
#         ca = OneDimensionalElementaryCellularAutomata(lattice_width=100)
#     p = density_from_configuration(configuration=ca.numpy())
#     if p not in density_frequency:
#         density_frequency[p] = 0 
#     density_frequency[p] += 1

#     for _ in range(100):
#         ca.transition(rule_number=3)
#     #imshow(ca.evolution(), cmap='grey')
#     #show()

#     theta_prior = 0.0
#     angles, derivatives = [],[]
#     for row in ca.evolution():
#         theta = angle(x=row)
#         angles.append(theta)
#         #derivatives.append(theta_d)
#         #derivatives.append(theta_prior - theta)
#         derivatives.append(theta_prior)
#         theta_prior = theta

#     all_angles.extend(angles)
#     all_derivatives.extend(derivatives)

#     #plot(angles)
#     #plot(derivatives)
#     plot(angles,derivatives,'-->')
# xlabel('theta')
# ylabel('dtheta/dt')
# show()

# bar(density_frequency.keys(), density_frequency.values())
# xlabel('density')
# ylabel('frequency')
# show()




# angle_directions = [
#     0.1*(a-b) for a,b in zip(all_angles[1:],all_angles[:-1])
# ]
# derivative_directions = [
#     0.1*(a-b) for a,b in zip(all_derivatives[1:],all_derivatives[:-1])
# ]
# colours = [
#     (abs(r),abs(g),(abs(r)+abs(g))/2) for r,g in zip(angle_directions, derivative_directions)
# ]

# quiver(
#     all_angles[:-1], all_derivatives[:-1], 
#     angle_directions, derivative_directions,
#     scale = 5,
#     color=colours
# )
# show()



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