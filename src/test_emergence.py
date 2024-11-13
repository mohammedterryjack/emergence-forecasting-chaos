from matplotlib.pyplot import imshow, show
from jpype import startJVM, getDefaultJVMPath
from numpy import array

from dynamical_system.eca.elementary_cellular_automata import ElementaryCellularAutomata
from metrics.emergence.transfer_entropy import TransferEntropy
from metrics.emergence.integrated_information import IntegratedInformation

#jvm_path = getDefaultJVMPath()
#print(jvm_path)
jvm_path = "/Users/mohammedterry-jack/Library/Java/JavaVirtualMachines/jdk-17.0.2+8/Contents/Home/lib/server/libjvm.dylib"
startJVM(jvm_path, f"-Djava.class.path=src/metrics/emergence/infodynamics.jar")

ca = ElementaryCellularAutomata(
    lattice_width=100,
    time_steps=100,
    transition_rule_number=30
) 

evolution = array(ca)
filtered_spacetime1 = TransferEntropy(k=7).emergence_filter(evolution=evolution)
filtered_spacetime2 = IntegratedInformation(k=7).emergence_filter(evolution=evolution)

imshow(evolution, cmap='gray')
show()
imshow(filtered_spacetime1)
show()
imshow(filtered_spacetime2)
show()