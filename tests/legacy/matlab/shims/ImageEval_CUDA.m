function V = ImageEval_CUDA(targets, sources, strengths, deltas)
% Shim: route the dead CUDA image kernel to the legacy CPU version.
V = ImageEval(targets, sources, strengths, deltas);
end
