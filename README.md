# emergence-forecasting-chaos
phd Thesis


## RUN:
```python

python src/matrixfactorisation_eca.py
python src/transformer_eca.py
```

## TODO:
[] update eca_and_emergence_encoder for all other files too
[] do for transformer eca like done for matfact eca

[] metrics complexity
[] metrics chaos

[] New experiment : Pretraining llm on more complex rules improves their ability to predict the next state for longer? Supported by this papers findings 

[] Alternative training setup which allows transformer to scale to any size EcA because its output state is always 2 now - instead of training it to predict the entire configuration - train the transformer to predict an individual cell state and then predict the configuration one cell at a time on the lattice. allow it N cells before and after as context. Then inspect attention mechanism to see if it uses only the minimal number of neighbours to learn the simplest rule possible or alternative solutions perhaps. Analyse this against the rule complexity. 
