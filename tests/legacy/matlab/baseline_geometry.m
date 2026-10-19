% Phase 1: legacy tower builders on fixed origins (for the geometry port).
p = legacy_setup();
hex = build_hex_tower([1 2 0.5]);
tri = build_tri_tower([0.5 -1 0]);
line = build_line_tower([3 1 0]);
save(fullfile(p.data, 'geometry.mat'), 'hex', 'tri', 'line', '-v7');
fprintf('geometry.mat written\n');
