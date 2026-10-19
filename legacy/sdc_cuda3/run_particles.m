%
% Detailed explanation goes here
%
%

%% 
% generate configuration file, towers.txt.
% one row in the file represents one tower: type x y z height level
% type = 0:hexagonal tower, 1:triangle, 2:line.
fprintf('generate towers.txt...\n');
% total_data = generate_hex_layout(2,20); % delta=2mm, h=20mm
total_data = generate_hex_layout(4,20); % delta=4mm, h=20mm
total_data=total_data'; % transpose for file output
fid=fopen('towers.txt','w');
fprintf(fid,'%f %f %f %f %f %f\n', total_data);
fclose(fid);

%%
fid=fopen('towers.txt');
X=fscanf(fid,'%f ',[6 inf]);
X=X';
fclose(fid);

%%
tower_number=size(X,1);
x_array=[];
for i=1:tower_number
    data=X(i,:);
    type=data(1);
    x=data(2);
    y=data(3);
    z=data(4);
    height=data(5);
    level=data(6);
    
%     type=2;
    
    if type==0 % hexagonal tower
        x_array = [x_array; build_hex_tower([x y z])];
    elseif type==1 % triangle tower
        x_array = [x_array; build_tri_tower([x y z])];
    elseif type==2 % line tower
        x_array = [x_array; build_line_tower([x y z])];
    end
end

%%
xsize = size(x_array,2)/3;
% flatten x,y,z into 1d array
x_component=x_array(:,1:xsize)';    
y_component=x_array(:,xsize+1:2*xsize)';
z_component=x_array(:,2*xsize+1:3*xsize)';
% concatenate x,y,z component
component_size = tower_number * xsize;
x0 = [reshape(x_component,1,component_size), ...
    reshape(y_component,1,component_size), ...
    reshape(z_component,1,component_size)];

%%
% compute gmres
fprintf('compute gmres...\n');
global sources;
global targets;
global deltas;
sources=x0';
targets=x0';
deltas=0.5;
Vel=sources;
% Vel(1:441)=Vel(883:1323)*-0.1;   % shear flow
% Vel(442:1323)=0;
Vel(1:component_size)=Vel(2*component_size+1:3*component_size)*-0.1;   % shear flow
Vel(component_size+1:3*component_size)=0;
[Forces]=gmres(@StokesSelf,Vel,40,1e-6);
global gForces;
gForces=reshape(Forces,component_size,3);

%%
fprintf('simulation begins...\n');
Tfin = 40.0;      %  Final run time
Dt = Tfin/100;   %  Time step (hex)
xend = particlesHex(x0,Tfin,Dt);
% xend = particles(x0,Tfin,Dt);

%xend = particles_image(x0,Tfin,Dt);

%  Make the exact solution
%x_exact = x0;
%x_exact(5) = x0(5)*exp(0.1*Tfin);
%x_exact(6) = x0(6)*exp(0.1*Tfin);

%  Compute error
%xend-x_exact
