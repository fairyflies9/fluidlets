function V = StokesletEval_CUDA(targets, sources, strengths, deltas)
% Shim: the legacy .mexw32 cannot load on 64-bit MATLAB, so route the
% CUDA call to the legacy CPU kernel, which computes the same sum.
V = StokesletEval(targets, sources, strengths, deltas);
end
