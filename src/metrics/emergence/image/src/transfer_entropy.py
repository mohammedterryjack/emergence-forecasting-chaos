"""Adapted from https://github.com/marcbrittain/Transfer_Entropy/blob/master/CAtransferEntropy.py"""

from enum import Enum 

from jpype import JArray, JInt, JPackage
from numpy import ndarray, roll, zeros

class TransferEntropyNeighbour(Enum):
    LEFT = 1
    RIGHT = -1

class TransferEntropy:
    def __init__(self, k:int=8, base:int=2) -> None:        
        self.metric = JPackage("infodynamics.measures.discrete").TransferEntropyCalculatorDiscrete(base,k)
        self.metric.initialise()

    def __call__(self, x:ndarray, y:ndarray) -> ndarray:
        x_ = JArray(JInt, 1)(x.tolist())
        y_ = JArray(JInt, 1)(y.tolist())
        self.metric.addObservations(y_,x_)
        return self.metric.computeLocalFromPreviousObservations(
            y_,
            x_
        )

def pointwise_transfer_entropy(
    evolution:ndarray, 
    neighbour:TransferEntropyNeighbour = TransferEntropyNeighbour.LEFT
) -> ndarray:   
    _,width = evolution.shape
    filtered_evolution = zeros(evolution.shape)
    emergence_filter = TransferEntropy()
    for column_index in range(width):
        filtered_evolution[:,column_index] = emergence_filter(
            x=evolution[:,column_index],
            y=roll(evolution,neighbour.value,1)[:,column_index]  
        )
    return filtered_evolution
