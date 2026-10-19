# Fluidlets

Regularized fundamental solutions for fluids and fields: the method of regularized Stokeslets, Brinkmanlets, and Oseenlets, regularized electrostatic kernels, and vortex sheets, with IB2d/IBAMR-compatible input files, VTK output, and a library of examples and Jupyter tutorials.

**Status: early development.** This 0.0.1 release contains only the 3D regularized Stokeslet (Cortez, Fauci & Medovikov 2005). The full package is being built at https://github.com/fairyflies9/fluidlets.

```python
import numpy as np
from fluidlets import stokeslet3d_velocity

targets = np.array([[1.0, 0.0, 0.0]])
sources = np.array([[0.0, 0.0, 0.0]])
forces  = np.array([[1.0, 0.0, 0.0]])
u = stokeslet3d_velocity(targets, sources, forces, eps=0.1, mu=1.0)
```

Developed by Laura A. Miller and collaborators, building on regularized Stokeslet code from the UNC glycocalyx project (2011).
