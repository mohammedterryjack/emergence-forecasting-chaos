# Dynamical Systems

## Cellular Automata
```python
eca = ElementaryCellularAutomata(
    neighbourhood_radius=1,
    transition_rule_number=110
)

eca.show()

eca.save('ca') #saves to file ca.txt

eca[0]
eca[10:]

list(ca)
array(ca)
```