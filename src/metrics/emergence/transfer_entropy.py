from jpype import JArray, JInt, JPackage
from numpy import ndarray, array

class TransferEntropy:
    def __init__(self, k:int) -> None:
        transfer_entropy = JPackage("infodynamics.measures.discrete").TransferEntropyCalculatorDiscrete
        self.metric = transfer_entropy(2,k)
        self.metric.initialise()

    def __call__(self, x:ndarray, y:ndarray) -> ndarray:
        x_ = JArray(JInt, 1)(x.tolist())
        y_ = JArray(JInt, 1)(y.tolist())
        self.metric.addObservations(y_,x_)
        result = self.metric.computeLocalFromPreviousObservations(
            y_,
            x_
        )
        return array(result)
