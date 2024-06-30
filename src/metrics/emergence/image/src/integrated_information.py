from jpype import JClass, JArray, JInt
from numpy import zeros, ndarray
from numpy import sum as np_sum


class IntegratedInformation:
    def __init__(
        self,
        phi_k:int,
        base:int = 2,  
        phi_width:int = 3
    ) -> None:
        self.phi_width = phi_width
        self.matrix_utils = JClass('infodynamics.utils.MatrixUtils')()
        self.metric3 = JClass('infodynamics.measures.discrete.ActiveInformationCalculatorDiscrete')(base**phi_width, phi_k)
        self.metric1 = JClass('infodynamics.measures.discrete.ActiveInformationCalculatorDiscrete')(base, phi_k)

    def __call__(self, train_evolution:ndarray, test_evolution:ndarray) -> ndarray:
        train_evolution_ = JArray(JInt, 1)(train_evolution.tolist())
        test_evolution_ = JArray(JInt, 1)(test_evolution.tolist())

        self.metric1.addObservations(train_evolution_)
        metric1_locals = self.metric1.computeLocalFromPreviousObservations(test_evolution_)

        _,width = train_evolution.shape
        _,test_width = test_evolution.shape

        for col in range(self.phi_width-1, width):
            aux = self.matrix_utils.computeCombinedValues(
                train_evolution[:, col-(self.phi_width-1):col+1], 
                self.base
            )
            aux_ = JArray(JInt, 1)(aux.tolist())
            self.metric3.addObservations(aux_)
        
        test_aux = zeros(test_evolution.shape)
        for col in range(self.phi_width-1, test_width):
            test_aux[:, col] = self.matrix_utils.computeCombinedValues(
                test_evolution[:, col-(self.phi_width-1):col+1], 
                self.base
            )
        
        test_aux_ = JArray(JInt, 1)(test_aux.tolist())
        metric3_locals = self.metric3.computeLocalFromPreviousObservations(test_aux_)

        R = (self.phi_width-1) // 2
        locals = zeros(test_evolution.shape)
        for col in range(R, test_width-R):
            locals[:, col] = metric3_locals[:, col] - np_sum(metric1_locals[:, col-R:col+R+1], axis=1)
        return locals
