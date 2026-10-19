% Phase 0, step 6: run the legacy glycocalyx driver end to end.
% run_particles.m builds the towers, solves GMRES for the forces that hold
% them fixed in the shear flow u = 0.1 z above a wall, then runs particlesHex,
% which writes the 32^3 velocity grid. Everything lands in the work folder.
p = legacy_setup();
t0 = tic;
run_particles;                                     % legacy script, unchanged
elapsed = toc(t0);

% Residual of the resistance solve, with the same matvec GMRES used.
global iter_count
iter_count = 0;
resid = norm(StokesSelf(Forces) - Vel) / norm(Vel);

fid = fopen(fullfile(p.work, 'velocity_t_xyz_final.raw'), 'r'); grid_new = fread(fid, 'float32'); fclose(fid);
fid = fopen(fullfile(p.legacy, 'velocity_t_xyz_final.raw'), 'r'); grid_old = fread(fid, 'float32'); fclose(fid);
grid_reldiff = norm(grid_new - grid_old) / norm(grid_old);

towers_file = fileread(fullfile(p.work, 'towers.txt'));
save(fullfile(p.data, 'towers.mat'), 'x0', 'Vel', 'Forces', 'xend', 'resid', ...
     'grid_new', 'grid_old', 'grid_reldiff', 'towers_file', 'elapsed', '-v7');
fprintf('towers.mat written: resid %.3e, grid reldiff vs stored .raw %.3e, %.0f s\n', resid, grid_reldiff, elapsed);
