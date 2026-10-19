x0 = build_hex_tower([0 0 0]);
global sources;
global targets;
global deltas;
sources=x0';
targets=x0';
deltas=0.5;
Vel=sources;
Vel(1:147)=Vel(295:441)*-0.1;   % shear flow
Vel(148:441)=0;
[Forces]=gmres(@StokesSelf,Vel,40,1e-6);
global gForces;
gForces=reshape(Forces,147,3);
