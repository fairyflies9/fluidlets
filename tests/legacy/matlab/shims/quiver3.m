function h = quiver3(varargin)
% Shim: the legacy plotme() passes a row vector of positions and a column
% vector of velocities, which current MATLAB rejects. Plots are not needed
% for the Phase 0 checks, so this shadows the built-in and draws nothing.
h = [];
end
