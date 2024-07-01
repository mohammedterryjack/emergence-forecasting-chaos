"""
Adapted from Integrated information as a common signature of dynamical and information-processing complexity (Pedro A M Mediano)
built upon Jidt:
https://github.com/jlizier/jidt/blob/master/java/source/infodynamics/measures/discrete/ActiveInformationCalculatorDiscrete.java
"""

from jpype import JClass, JArray, JInt
from numpy import zeros, ndarray, array
from numpy import sum as np_sum

class IntegratedInformation:
    def __init__(
        self,
        k:int,
        base:int = 2,  
        phi_width:int = 3
    ) -> None:
        self.base = base
        self.phi_width = phi_width
        self.matrix_utils = JClass('infodynamics.utils.MatrixUtils')()
        self.metric3 = JClass('infodynamics.measures.discrete.ActiveInformationCalculatorDiscrete')(self.base**self.phi_width, k)
        self.metric1 = JClass('infodynamics.measures.discrete.ActiveInformationCalculatorDiscrete')(self.base, k)

    def _calculate(self, train_evolution:ndarray, test_evolution:ndarray) -> ndarray:
        self.metric1.addObservations(
            JArray(JArray(JInt))(train_evolution.tolist())
        )
        metric1_locals = array(self.metric1.computeLocalFromPreviousObservations(
            JArray(JArray(JInt))(test_evolution.tolist())
        ))

        _,width = train_evolution.shape
        _,test_width = test_evolution.shape

        for col in range(self.phi_width-1, width):
            aux = self.matrix_utils.computeCombinedValues(
                JArray(JArray(JInt))(train_evolution[:, col-(self.phi_width-1):col+1]), 
                self.base
            )
            self.metric3.addObservations(aux)
        
        test_aux = zeros(test_evolution.shape, dtype=int)
        for col in range(self.phi_width-1, test_width):
            test_aux[:, col] = self.matrix_utils.computeCombinedValues(
                JArray(JArray(JInt))(test_evolution[:, col-(self.phi_width-1):col+1]), 
                self.base
            )
        metric3_locals = array(self.metric3.computeLocalFromPreviousObservations(
            JArray(JArray(JInt))(test_aux.tolist())
        ))

        R = (self.phi_width-1) // 2
        locals = zeros(test_evolution.shape)
        for col in range(R, test_width-R):
            locals[:, col] = metric3_locals[:, col] - np_sum(metric1_locals[:, col-R:col+R+1], axis=1)
        return locals

    def emergence_filter(self, evolution:ndarray) -> ndarray:
        return self._calculate(train_evolution=evolution, test_evolution=evolution)
    