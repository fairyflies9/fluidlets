# Tutorials

Jupyter notebooks that teach the method of regularized Stokeslets step by step, with the math
next to the code. Each one ends with exercises. They run top to bottom in under a minute on a
laptop, and on Colab or Binder (the first code cell installs Fluidlets if it is missing).

| | notebook | what you learn |
|---|---|---|
| 01 | [From blob to regularized Stokeslet](01_blob_to_stokeslet.ipynb) | derive $H_1$, $H_2$ from the blob; compare with the singular Stokeslet; $O(\varepsilon^2)$ convergence |
| 02 | [Choosing ε](02_choosing_eps.ipynb) | sphere torque vs ε and point spacing; regularization vs discretization error; ε ≈ h/3 |
| 03 | [Mobility, resistance, and swimming](03_mobility_resistance_swimming.ipynb) | `evaluate` vs `resistance`; mixed problems; walls vs Brenner; free swimming |

Run them all as tests with

```
python -m pytest --nbmake tutorials
```

(needs `jupyter`, `matplotlib` and `nbmake`). Instructor solutions to the exercises belong in
`courses/*/instructor/`, never in these notebooks.
