"""Adapted from https://github.com/marcbrittain/Transfer_Entropy/blob/master/CAtransferEntropy.py"""

from enum import Enum 

from jpype import JArray, JInt, JPackage, startJVM, getDefaultJVMPath
from numpy import ndarray, roll, zeros

startJVM(getDefaultJVMPath(), f"-Djava.class.path=/app/infodynamics.jar")

class TransferEntropy:
    def __init__(self, k:int) -> None:
        transfer_entropy = JPackage("infodynamics.measures.discrete").TransferEntropyCalculatorDiscrete
        self.metric = transfer_entropy(2,k)
        self.metric.initialise()

    def __call__(self, x:ndarray, y:ndarray) -> ndarray:
        x_ = JArray(JInt, 1)(x.tolist())
        y_ = JArray(JInt, 1)(y.tolist())
        self.metric.addObservations(y_,x_)
        return self.metric.computeLocalFromPreviousObservations(
            y_,
            x_
        )

class TransferEntropyNeighbour(Enum):
    LEFT = 1
    RIGHT = -1

def pointwise_transfer_entropy(
    evolution:ndarray, 
    k_history:int, 
    neighbour:TransferEntropyNeighbour
) -> ndarray:   
    metric = TransferEntropy(k=k_history)
    _,width = evolution.shape
    filtered_evolution = zeros(evolution.shape)
    for column_index in range(width):
        filtered_evolution[:,column_index] = metric(
            x=evolution[:,column_index],
            y=roll(evolution,neighbour.value,1)[:,column_index]  
        )
    return filtered_evolution
