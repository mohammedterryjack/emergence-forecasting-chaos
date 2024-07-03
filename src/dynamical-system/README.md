# Dynamical Systems

## Cellular Automata
```python
ca = OneDimensionalBinaryCellularAutomata(
    neighbourhood_radius=1,
    transition_rule_number=110
)

ca.show() #plots ca

ca.save('ca') #saves to file ca.txt

ca[0]
ca[10:]

list(ca)
array(ca)
```