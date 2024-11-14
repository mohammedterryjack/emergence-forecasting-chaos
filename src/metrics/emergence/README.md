# Example

```python
from numpy import array

from metrics.emergence import TransferEntropy, IntegratedInformation

ca = ElementaryCellularAutomata(
    lattice_width=100,
    time_steps=100,
    transition_rule_number=30
)
evolution = array(ca)
```

```python
from dynamical_system.eca.elementary_cellular_automata import ElementaryCellularAutomata

filtered_spacetime1 = TransferEntropy(k=7).emergence_filter(evolution=evolution)
filtered_spacetime2 = IntegratedInformation(k=7).emergence_filter(evolution=evolution)
```