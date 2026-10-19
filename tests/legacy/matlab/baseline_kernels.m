% Phase 0, steps 2-3: legacy Stokeslet and image kernels on fixed inputs.
p = legacy_setup();
rng(20111);
ns = 40; nt = 60;
sources = [rand(ns,2)*2-1, 0.2 + rand(ns,1)];     % above the wall z = 0
targets = [rand(nt,2)*3-1.5, 0.05 + 1.5*rand(nt,1)];
strengths = randn(ns,3);
deltas = 0.05 + 0.2*rand(1,ns);                    % per-source blob widths

V_stokeslet = StokesletEval(targets, sources, strengths, deltas);
V_image     = ImageEval(targets, sources, strengths, deltas);

% Velocity on the wall itself: should vanish for a no-slip image system.
[gx, gy] = meshgrid(linspace(-2,2,21));
wall = [gx(:), gy(:), zeros(numel(gx),1)];
V_wall_total = StokesletEval(wall, sources, strengths, deltas) ...
             + ImageEval(wall, sources, strengths, deltas);
V_wall_free  = StokesletEval(wall, sources, strengths, deltas);

% Wall error vs blob size for one Stokeslet at height 1 (each force direction).
eps_list = [0.4 0.2 0.1 0.05 0.025];
wall_err = zeros(numel(eps_list), 3);
for k = 1:numel(eps_list)
  for c = 1:3
    f = zeros(1,3); f(c) = 1;
    u = StokesletEval(wall, [0 0 1], f, eps_list(k)) + ImageEval(wall, [0 0 1], f, eps_list(k));
    wall_err(k,c) = max(abs(u(:)));
  end
end

save(fullfile(p.data, 'kernels.mat'), 'sources', 'targets', 'strengths', 'deltas', ...
     'V_stokeslet', 'V_image', 'wall', 'V_wall_total', 'V_wall_free', 'eps_list', 'wall_err', '-v7');
fprintf('kernels.mat written\n');
