"""Phase 0: checks of the 2011 legacy code (legacy/sdc_cuda3).

The .mat files in tests/legacy/data are written by the MATLAB drivers in
tests/legacy/matlab (run with MATLAB R2024a; see docs/legacy_verification.md).
Each test is skipped if its data file has not been generated.
"""
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from fluidlets import stokeslet3d_velocity
from legacy_kernels import blake, image_eval

DATA = Path(__file__).parent / "data"


def load(name):
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{name} not generated; run tests/legacy/matlab/{name[:-4]}*.m")
    return sio.loadmat(path, squeeze_me=True)


# --- Step 2: the legacy Stokeslet is the Cortez-Fauci-Medovikov kernel -------

def test_legacy_stokeslet_equals_cfm():
    d = load("kernels.mat")
    u = stokeslet3d_velocity(d["targets"], d["sources"], d["strengths"], d["deltas"])
    assert np.allclose(u, d["V_stokeslet"], rtol=1e-13, atol=1e-15)


# --- Step 3: the image system ------------------------------------------------

def test_image_transcription_matches_matlab():
    d = load("kernels.mat")
    u = image_eval(d["targets"], d["sources"], d["strengths"], d["deltas"])
    assert np.allclose(u, d["V_image"], rtol=1e-13, atol=1e-15)


def test_images_give_zero_velocity_on_wall():
    d = load("kernels.mat")
    assert np.abs(d["V_wall_free"]).max() > 0.1          # the wall matters here
    assert np.abs(d["V_wall_total"]).max() < 1e-14
    assert d["wall_err"].max() < 1e-15                   # for every eps and force direction


def wall_velocity(x, x0, f, eps):
    return (stokeslet3d_velocity([x], [x0], [f], eps) + image_eval([x], [x0], [f], eps))[0]


def test_wall_system_is_divergence_free():
    x0 = np.array([0.3, -0.2, 1.0])
    rng = np.random.default_rng(1)
    h = 1e-4
    for _ in range(3):
        x = rng.uniform([-1, -1, 0.2], [1, 1, 2])
        f = rng.normal(size=3)
        div = sum(
            (wall_velocity(x + h * e, x0, f, 0.05)[i] - wall_velocity(x - h * e, x0, f, 0.05)[i]) / (2 * h)
            for i, e in enumerate(np.eye(3))
        )
        assert abs(div) < 1e-6


def test_wall_system_converges_to_blake_at_second_order():
    x0 = np.array([0.3, -0.2, 1.0])
    x = np.array([1.2, 0.4, 0.6])
    f = np.array([0.3, -1.0, 0.7])
    exact = blake(x, x0, f)
    eps = np.array([0.1, 0.05, 0.025, 0.0125])
    err = [np.abs(wall_velocity(x, x0, f, e) - exact).max() for e in eps]
    rates = np.log2(np.array(err[:-1]) / np.array(err[1:]))
    assert np.all(rates > 1.9)


def test_blake_reference_is_zero_on_wall():
    # guards the transcription of Blake's formula used above
    x0 = np.array([0.3, -0.2, 1.0])
    for f in np.eye(3):
        assert np.abs(blake([1.1, 0.7, 0.0], x0, f)).max() < 1e-10


# --- Step 4: resistance solve on a sphere ------------------------------------

def test_sphere_drag_converges_to_stokes():
    d = load("sphere_drag.mat")
    rel = d["F_free"] / d["F_stokes"] - 1
    assert np.all(np.diff(rel) < 0)              # monotone approach with refinement
    assert 0 < rel[-1] < 0.015                   # N = 800, eps = spacing/2


def test_sphere_drag_near_wall_matches_reflections():
    d = load("sphere_drag.mat")
    s = 1 / d["H_list"]
    faxen = 1 / (1 - 9 / 16 * s + s**3 / 8 - 45 / 256 * s**4 - s**5 / 16)
    lorentz = 1 / (1 - 9 / 8 * s + s**3 / 2)
    par = d["F_par"] / d["F_free_N"]
    perp = d["F_perp"] / d["F_free_N"]
    assert np.allclose(par, faxen, rtol=0.005)
    # The two-term perpendicular series is poor closer in (1.8% off at H/a = 3);
    # checking H/a < 5 needs Brenner's (1961) exact values.
    far = d["H_list"] >= 5
    assert np.allclose(perp[far], lorentz[far], rtol=0.01)


# --- Step 5: the legacy SDC driver ------------------------------------------

def test_legacy_sdc_is_first_order():
    # Documents a legacy bug (lagged spring forces, stale fdiff); see
    # docs/legacy_verification.md. Nominal order is 9.
    d = load("sdc_order.mat")
    assert np.all(np.abs(d["order"] - 1) < 0.1)


# --- Step 6: glycocalyx tower baseline ---------------------------------------

LEGACY = Path(__file__).parents[2] / "legacy" / "sdc_cuda3"


def test_tower_rerun_reproduces_stored_cuda_grid():
    d = load("towers.mat")
    stored = np.fromfile(LEGACY / "velocity_t_xyz_final.raw", "<f4")
    rerun = d["grid_new"].astype(np.float64)
    assert np.linalg.norm(rerun - stored) / np.linalg.norm(stored) < 1e-6   # float32 CUDA


def test_towers_held_fixed_by_gmres_forces():
    d = load("towers.mat")
    assert d["resid"] < 1e-6
    assert np.array_equal(d["xend"], d["x0"])          # particlesHex never moves them
    n = d["x0"].size // 3
    X = d["x0"].reshape(3, n).T
    F = d["Forces"].reshape(3, n).T
    on_wall = X[:, 2] == 0
    assert np.abs(F[on_wall]).max() < 1e-10            # undetermined, left at GMRES's zero guess
    drag = [F[k * 147:(k + 1) * 147, 0].sum() for k in range(3)]
    assert np.isclose(drag[0], drag[2], rtol=1e-6)     # outer towers mirror each other
    assert abs(drag[1]) < abs(drag[0])                 # middle tower is shielded
