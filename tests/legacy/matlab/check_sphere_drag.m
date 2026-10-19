% Phase 0, step 4: resistance problem for a translating sphere with the
% legacy kernels. Free space: drag vs 6*pi*mu*a*U (mu = 1 in the legacy
% code). Near the wall: drag vs the classical method-of-reflections series.
p = legacy_setup();
a = 1;

% Free space, refining the surface, eps = 0.5 * spacing... and fixed eps.
N_list = [100 200 400 800];
F_free = zeros(numel(N_list), 1); h_list = zeros(numel(N_list),1);
for k = 1:numel(N_list)
  X = fib_sphere(N_list(k), a, [0 0 0]);
  h = sqrt(4*pi*a^2 / N_list(k)); h_list(k) = h;
  F = resist(X, [1 0 0], 0.5*h, false);
  F_free(k) = sum(F(:,1));
end
F_stokes = 6*pi*a;           % mu = 1

% Wall: sphere centre at height H, N = 400, eps = 0.5 * spacing.
H_list = [1.5 2 3 5 10];
N = 400; h = sqrt(4*pi*a^2/N); epsw = 0.5*h;
F_par = zeros(numel(H_list),1); F_perp = zeros(numel(H_list),1);
for k = 1:numel(H_list)
  X = fib_sphere(N, a, [0 0 H_list(k)]);
  F = resist(X, [1 0 0], epsw, true); F_par(k)  = sum(F(:,1));
  F = resist(X, [0 0 1], epsw, true); F_perp(k) = sum(F(:,3));
end
X = fib_sphere(N, a, [0 0 0]); F = resist(X, [1 0 0], epsw, false); F_free_N = sum(F(:,1));

save(fullfile(p.data, 'sphere_drag.mat'), 'a', 'N_list', 'h_list', 'F_free', 'F_stokes', ...
     'H_list', 'F_par', 'F_perp', 'F_free_N', 'epsw', '-v7');
fprintf('sphere_drag.mat written\n');

function F = resist(X, U, epsb, wall)
% Solve sum_k G(x_j, x_k) f_k = U at every surface point with GMRES,
% using the legacy CPU kernels as the matvec (same as StokesSelf).
n = size(X,1);
rhs = repmat(U, n, 1); rhs = rhs(:);
mv = @(f) apply(X, reshape(f, n, 3), epsb, wall);
[f, flag] = gmres(mv, rhs, 60, 1e-10, 40);
if flag ~= 0, warning('gmres flag %d', flag); end
F = reshape(f, n, 3);
end

function v = apply(X, f, epsb, wall)
V = StokesletEval(X, X, f, epsb);
if wall, V = V + ImageEval(X, X, f, epsb); end
v = V(:);
end

function X = fib_sphere(n, a, c)
k = (0:n-1)' + 0.5;
phi = acos(1 - 2*k/n); th = pi*(1 + sqrt(5))*k;
X = a*[cos(th).*sin(phi), sin(th).*sin(phi), cos(phi)] + c;
end
