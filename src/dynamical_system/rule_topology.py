from numpy.linalg import norm
from numpy import arccos, ones, ndarray, array
from matplotlib.pyplot import quiver, plot, show, annotate, title, xlabel, ylabel

from elementary_cellular_automata import ElementaryCellularAutomata

def convert_zeros_to_minus_one(x:ndarray) -> ndarray:
    return 2*x - 1

def parity(x:ndarray) -> float:
    return abs(sum(convert_zeros_to_minus_one(x)))


def cosine_similarity(a:ndarray, b:ndarray) -> float:
    result = a @ b.T
    result /= (norm(a)*norm(b))+1e-9
    return result

def angle(x:ndarray, origin:ndarray) -> float:
    cos_theta = cosine_similarity(a=origin,b=convert_zeros_to_minus_one(x))
    return arccos(cos_theta)

rule = 3
width = 16
display_configurations = False
display_all_trajectories = False
display_single_trajectory = True

ref_point_b = ones(shape=(width))

if display_configurations:
    seen = set()
    ca = ElementaryCellularAutomata()
    for ic in range(2**width):
        config = ca.create_binary_lattice_from_number(
            state_number=ic,
            lattice_width=width
        )
        x = parity(x=array(config))
        y = angle(x=array(config),origin=ref_point_b)
        while (x,y) in seen:
            y += 0.03
        seen.add((x,y))
        annotate(
            ca.stringify_configuration(
                configuration=config,
                representation_zero=ca.repr_zero,
                representation_one=ca.repr_one
            ), 
            xy=(x,y),
            fontsize=3
        )


if display_all_trajectories:
    T = 2
    for ic in range(2**width):
        ca =  ElementaryCellularAutomata(
            lattice_width=width,
            initial_state= ic,
            time_steps=T,
            transition_rule_number=rule
        )
        ys = list(map(lambda x:angle(x=x,origin=ref_point_b),ca))
        xs = list(map(lambda x:parity(x=x),ca))
        plot(xs,ys,'-->',color='orange', linewidth=1)
        #us = [a-b for a,b in zip(xs[1:],xs[:-1])]
        #vs = [a-b for a,b in zip(ys[1:],ys[:-1])]        
        #quiver(xs[:-1],ys[:-1],us,vs, color='orange')

if display_single_trajectory:
    T = 100
    ca =  ElementaryCellularAutomata(
        lattice_width=width,
        time_steps=T,
        transition_rule_number=rule
    )
    ys = list(map(lambda x:angle(x=x,origin=ref_point_b),ca))
    xs = list(map(lambda x:parity(x=x),ca))
    plot(xs,ys,'-->',color='red', linewidth=1)

print(ca)
title(rule)
xlabel('parity')
ylabel('whiteness (angle)')
show()

zs = [a+b for a,b in zip(xs,ys)]

plot(zs[1:],zs[:-1],'-->')
xlabel('t')
ylabel('t+1')
show()
