from jpype import JClass
from numpy import zeros, ndarray
from numpy import sum as np_sum

from eca import OneDimensionalElementaryCellularAutomata


def compute_phi(evolution:ndarray, phi_K:int) -> ndarray:
    # Initialize the variables
    base = 2  
    phi_width = 3

    matrix_utils = JClass('infodynamics.utils.MatrixUtils')()

    width = X.shape[1]
    testWidth = test.shape[1]

    # aisCalc1 = javaObject('infodynamics.measures.discrete.ActiveInformationCalculatorDiscrete', base, phi_K);
    # aisCalc3 = JPackage("infodynamics.measures.discrete").ActiveInformationCalculatorDiscrete
    # aisCalc3 = javaObject('infodynamics.measures.discrete.ActiveInformationCalculatorDiscrete', base^phi_width, phi_K);
    ActiveInformationCalculatorDiscrete = JClass('infodynamics.measures.discrete.ActiveInformationCalculatorDiscrete')
    aisCalc3 = ActiveInformationCalculatorDiscrete(base**phi_width, phi_K)
    aisCalc1 = ActiveInformationCalculatorDiscrete(base, phi_K)


    # Add training data to build the probability distributions

    # for col=phi_width:width
    for col in range(phi_width-1, width):
        #aux = mu.computeCombinedValues(X(:,col-(phi_width-1):col), base);
        aux = matrix_utils.computeCombinedValues(X[:, col-(phi_width-1):col+1], base)
        aisCalc3.addObservations(aux)
    aisCalc1.addObservations(X)



    # Initialize variables to calculate Phi
    phi_locals = zeros(test.shape)
    ais3_locals = zeros(test.shape)
    ais1_locals = zeros(test.shape)

    test_aux = zeros(test.shape)
    R = (phi_width-1) // 2

    # for col=phi_width:testWidth
    for col in range(phi_width-1, testWidth):
        #test_aux(:,col-1) = mu.computeCombinedValues(test(:,col-(phi_width-1):col), base);
        test_aux[:, col] = matrix_utils.computeCombinedValues(test[:, col-(phi_width-1):col+1], base)

    ais3_locals = aisCalc3.computeLocalFromPreviousObservations(test_aux)
    ais1_locals = aisCalc1.computeLocalFromPreviousObservations(test)


    # for col=1+R:testWidth-R
    for col in range(R, testWidth-R):
        #phi_locals(:,col) = ais3_locals(:,col) - sum(ais1_locals(:, col-R:col+R), 2);
        phi_locals[:, col] = ais3_locals[:, col] - np_sum(ais1_locals[:, col-R:col+R+1], axis=1)


    return phi_locals


ca =  OneDimensionalElementaryCellularAutomata(lattice_width=10000)
for _ in range(630):
    ca.transition(rule_number=54)
filtered_evolution_54 = compute_phi(
    evolution=ca.evolution(),
    phi_K = 4
)

ca =  OneDimensionalElementaryCellularAutomata(lattice_width=10000)
for _ in range(630):
    ca.transition(rule_number=110)
filtered_evolution_110 = compute_phi(
    evolution=ca.evolution(),
    phi_K = 5 
)
