% Phase 0, step 5: measured order of accuracy of the legacy SDC driver
% (particles.m: deformable triangle tower with springs in shear flow).
% No exact solution, so use self-convergence: halve Dt and compare final
% positions. Nominal order with 9 nodes and 9 sweeps is 9.
p = legacy_setup();
x0 = build_tri_tower([0 0 0]);
Tfin = 0.4;
nsteps = [2 4 8 16 32];
xend = zeros(numel(nsteps), numel(x0));
for k = 1:numel(nsteps)
  xend(k,:) = particles(x0, Tfin, Tfin/nsteps(k));
  close all
end
err = vecnorm(diff(xend), 2, 2);           % ||x(Dt) - x(Dt/2)||
order = log2(err(1:end-1) ./ err(2:end));
save(fullfile(p.data, 'sdc_order.mat'), 'x0', 'Tfin', 'nsteps', 'xend', 'err', 'order', '-v7');
fprintf('sdc_order.mat written; observed orders: %s\n', mat2str(order', 3));
