"""
Adapted from https://github.com/marcbrittain/Transfer_Entropy/blob/master/CAtransferEntropy.py
build upon Jidt:
https://github.com/jlizier/jidt/blob/master/java/source/infodynamics/measures/discrete/TransferEntropyCalculatorDiscrete.java
"""

from enum import Enum 

from jpype import JArray, JInt, JPackage
from numpy import ndarray, roll, zeros

class TransferEntropyNeighbour(Enum):
    LEFT = 1
    RIGHT = -1

class TransferEntropy:
    def __init__(self, k:int, base:int=2, neighbour:TransferEntropyNeighbour = TransferEntropyNeighbour.LEFT) -> None:        
        self.metric = JPackage("infodynamics.measures.discrete").TransferEntropyCalculatorDiscrete(base,k)
        self.metric.initialise()
        self.neighbour = neighbour

    def _calculate(self, x:ndarray, y:ndarray) -> ndarray:
        x_ = JArray(JInt, 1)(x.tolist())
        y_ = JArray(JInt, 1)(y.tolist())
        self.metric.addObservations(y_,x_)
        return self.metric.computeLocalFromPreviousObservations(
            y_,
            x_
        )

    def emergence_filter(
        self,
        evolution:ndarray, 
    ) -> ndarray:   
        _,width = evolution.shape
        filtered_evolution = zeros(evolution.shape)
        for column_index in range(width):
            filtered_evolution[:,column_index] = self._calculate(
                x=evolution[:,column_index],
                y=roll(evolution,self.neighbour.value,1)[:,column_index]  
            )
        return filtered_evolution
