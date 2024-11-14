# emergence-forecasting-chaos
phd Thesis


## RUN:
```python

python src/matrixfactorisation_eca.py
python src/transformer_eca.py
```

## TODO:
[x] emergent metrics working
[x] add path to settings and environ variables etc
[x] update emergence metric to take in one row, but also the context for some timesteps back (if not enough context, set k appropriately - or set to 0s for those)
[x] include emergent features metrics to utils and then model predictions
[] update eca_and_emergence_encoder for all other files too
[] metrics complexity
[] metrics chaos