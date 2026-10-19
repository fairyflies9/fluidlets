"""Exact Stokes solutions used for validation, tests and examples."""
from __future__ import annotations

import numpy as np


def translating_sphere_velocity(x, U, a: float = 1.0) -> np.ndarray:
    """Velocity outside a sphere of radius a, centred at 0, translating with U.

    u = (3a/4)(U/r + (U.x)x/r^3) + (a^3/4)(U/r^3 - 3(U.x)x/r^5)
    (e.g. Cortez, Fauci & Medovikov 2005, Eq. 24). Inside (r < a), u = U.
    """
    x = np.atleast_2d(np.asarray(x, float))
    U = np.asarray(U, float)
    r_true = np.linalg.norm(x, axis=1)[:, None]
    r = np.maximum(r_true, a)            # the formula is only used outside; avoids 0/0 at the centre
    Ux = (x @ U)[:, None]
    u = 0.75 * a * (U / r + Ux * x / r**3) + 0.25 * a**3 * (U / r**3 - 3 * Ux * x / r**5)
    return np.where(r_true < a, U, u)


def brenner_wall_drag(d_over_a, terms: int = 400) -> np.ndarray:
    """Drag on a sphere moving perpendicular to a plane wall, over 6 pi mu a U.

    Brenner (1961), Chem. Eng. Sci. 16, 242, as quoted in Ainley et al.
    (2008), Eq. 22, with alpha = arccosh(d/a), d the centre-to-wall distance:

        F* = (4/3) sinh(alpha) sum_{n>=1} n(n+1)/((2n-1)(2n+3))
             [ (2 sinh((2n+1)alpha) + (2n+1) sinh(2 alpha))
               / (4 sinh^2((n+1/2)alpha) - (2n+1)^2 sinh^2(alpha)) - 1 ].

    The series converges slowly as d/a -> 1; Ainley et al. use it for d/a >= 1.02.
    """
    alpha = np.arccosh(np.atleast_1d(np.asarray(d_over_a, float)))[:, None]
    n = np.arange(1, terms + 1)[None, :]
    with np.errstate(over="ignore", invalid="ignore"):
        num = 2 * np.sinh((2 * n + 1) * alpha) + (2 * n + 1) * np.sinh(2 * alpha)
        den = 4 * np.sinh((n + 0.5) * alpha) ** 2 - (2 * n + 1) ** 2 * np.sinh(alpha) ** 2
        term = n * (n + 1) / ((2 * n - 1) * (2 * n + 3)) * (num / den - 1)
    term = np.where(np.isfinite(term), term, 0.0)      # far terms overflow to 0/inf; they vanish
    return (4 / 3) * np.sinh(alpha[:, 0]) * term.sum(axis=1)


def cooley_oneill_wall_drag(d_over_a) -> np.ndarray:
    """Near-contact asymptotics for perpendicular approach (Cooley & O'Neill 1969).

    F* ~ 1/xi - (1/5) ln(xi) + 0.971280, xi = d/a - 1 (Ainley et al. 2008, Eq. 23).
    """
    xi = np.asarray(d_over_a, float) - 1
    return 1 / xi - 0.2 * np.log(xi) + 0.971280
