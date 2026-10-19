% x0 = build_hex_tower([0 0 0]);

%  build 3 hex towers
delta=2;
clear x_array;
x_array = build_line_tower([0 0 0]);
% x_array = [x_array; build_hex_tower([sqrt(3)*delta           0 0])];
% x_array = [x_array; build_hex_tower([sqrt(3)*delta/2 3/2*delta 0])];

% x_array = [x_array; build_hex_tower([0 2 0])];
% x_array = [x_array; build_hex_tower([0 4 0])];

x_array = [x_array; build_line_tower([0 4 0])];
x_array = [x_array; build_line_tower([0 8 0])];

xsize = size(x_array,2)/3;
x0 = [x_array(1,1:xsize) x_array(2,1:xsize) x_array(3,1:xsize) ...
    x_array(1,xsize+1:2*xsize) x_array(2,xsize+1:2*xsize) x_array(3,xsize+1:2*xsize)...
    x_array(1,2*xsize+1:3*xsize) x_array(2,2*xsize+1:3*xsize) x_array(3,2*xsize+1:3*xsize)];

global sources;
global targets;
global deltas;
sources=x0';
targets=x0';
deltas=0.5;
Vel=sources;
Vel(1:63)=Vel(127:189)*-0.1;   % shear flow
Vel(64:189)=0;
[Forces]=gmres(@StokesSelf,Vel,40,1e-6);
global gForces;
gForces=reshape(Forces,63,3);
