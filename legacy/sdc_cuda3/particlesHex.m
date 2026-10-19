function [xend]=particlesHex(x0,Tfin,bigDt)
%
%  Code to demonstrate the SDC method on a particle system
%
%                x_t  = v(x,t)
%
%  Here x is a vector of (time-dependent) positions
%
%  The method used is an explicit SDC method.
%
%  The order of the method can be changed by setting the variables
%  sdcorder and niter below.
%
%  This purpose of this code is strictly  pedagogical, and the
%  author makes no claims regarding the efficiency, accuracy,
%  appropriateness, style, or correctness of this file.
%
%  Michael L. Minion
%  Copyright Oct. 2005.


%-----------------------------------------------
%  Problem set up
%----------------------------------------------
figure(3);
clf;
xend=[];
nbigsteps = round(Tfin/bigDt);       %  Number of time steps
if (abs(nbigsteps*bigDt-Tfin)>1e-10)
    display('bad bigDt')
    return;
end

[n] = length(x0);           % number of  unknowns
npart = n/3;                % number of particles


%-----------------------------------------------
%  Make integration matrices
%----------------------------------------------
sdcorder = 9;                 % Number of quadrature nodes
niter = 2;                    % Total number of iterations (order of method)
whichpts = 2;                 % Type of points to use (see makesig.m)
Lopt = 0;                     % Left endpoint rule  (see makesig.m)
sigint = makesig(sdcorder,whichpts,Lopt);   %  Integration matrices
sigmate = squeeze(sigint(:,:,1))*bigDt;     %  Explicit rule
sigmati = squeeze(sigint(:,:,2))*bigDt;     %  Implicit rule (not used)
dtsdc = sum(sigmate');                      %  Time substeps
nsdc = length(dtsdc)+1;                     %  Number of  substeps

%-----------------------------------------------
%  Make space for solution and other quantities
%----------------------------------------------
x = zeros(nbigsteps,3*npart,nsdc);   %  Solution at all times
fe = zeros(3*npart,nsdc);            %  Explicit function values
fi = zeros(3*npart,nsdc);            %  Explicit function values
fsave = zeros(3*npart,1);              %  Temp space for fe
fdiff = zeros(3*npart,1);              %  Difference of fe in corr. step
Immf = zeros(3*npart,nsdc-1);        %  Time integrals in corr. step

%-----------------------------------------------------
display('----------   Computing solution ---------- ');
%-----------------------------------------------------
display(['Taking ',num2str(nbigsteps),' time steps with Dt=',num2str(bigDt)]);
display(['Doing ',num2str(niter),' total iterations']);

%-----------------------------------------------
%  Initialize some stuff
%----------------------------------------------

% tracing particles to visualize the flow
% target_size=10;
% target = zeros(target_size * target_size,3);
% target_width=0.05;
% target_height=1;
% count=1;
% for i=0:target_size-1
%     for j=0:target_size-1
%         target(count,2)=-target_width + i/(target_size-1) * 2 * target_width;
%         target(count,3)=j/(target_size-1) * target_height;
%         count = count + 1;
%     end
% end

% one line of tracing particles.
tracing_size=64;
% for one hex tower
% target_y=linspace( -2,  2,tracing_size)';
% for 0,2,4, hex towers
% target_y=linspace( -2,  6,tracing_size)';
% for 0,4,8, hex towers
target_y=linspace( -2,  10,tracing_size)';

target_x=-1.5 * ones(tracing_size,1);
target_z=6 * ones(tracing_size,1);
tracing = [target_x target_y target_z];



%%
%fenew=compfe(x0,0);       % Compute initial explicit  function
%fenew=comp_stokeslets(x0,0);

num = size(x0,2)/3;

strengths = zeros(num,3);
% deltas = 0.1;   % for triangle, 0.1 needs 600 steps.
deltas = 0.5;

fenew = zeros(1,num*3);
% convert data structure:(1,3*num)->(num,3)
sources=reshape(x0,num,3);

% compute velocity
V=StokesletCUDA(sources,sources,strengths,deltas);

% convert data structure back:(num,3)->(1,3*num)
fenew=V(:)';
% for initial value, there is no background flow.

fe(:,1)=squeeze(fenew);   % Store it
x(1,:,1)=x0;              % Store initial condition

% compute initial length as resting length.
edge = sources(1,:) - sources(2,:);
resting_length = norm(edge);
% vertical edge
ver_edge = sources(1,:) - sources(8,:);
vertical_length = norm(ver_edge);
% diag edge
diag_edge = sources(1,:) - sources(9,:);
diagonal_length = norm(diag_edge);
        
global gForces;
%-----------------------------------------------
%  Main time step loop
%----------------------------------------------
%%
% visualize plane tracing particles

%%
filename='velocity_t_xyz_final.raw';
fprintf('output velocity grid to %s...\n', filename);

% compute_velocity_grid(filename,sources, 32, -1.6, 1.6, -2, 10, 0, 10)
compute_velocity_grid(filename,sources, 32, -1, 8, -5, 25, 0, 20)

fprintf('done.\n');
% return;

%%
for k0=1:nbigsteps               %  Time step loop
    tic;
    fprintf('step %2d/%2d, ',k0, nbigsteps);
    
    tsdc = (k0-1)*bigDt;           %  Time at beginning of time step
    if(k0 > 1)
        x(k0,:,1)=x(k0-1,:,end);     %  Resetting solution
        fe(:,1)=fe(:,end);           %  Resetting explicit function
    end
    
    % visualize iniital state
    if (k0 == 1)
        plotme(x(k0,:,1),dtsdc,nbigsteps,fe(:,1),strengths,tracing,k0,deltas) 
    end
    
    %  Predictor loop
    for mk = 1:nsdc-1              % begin predictor loop
        %    display(['predictor k0=',num2str(k0),' mk=',num2str(mk)]);
        
        % udpate position
        xnew = squeeze(x(k0,:,mk)) + dtsdc(mk)*fe(:,mk)';  %  Forward Euler step
        tsdc = tsdc+dtsdc(mk);                             %  Update time
        
        % original simple evaluation
        %[fenew] = compfe(xnew,tsdc);       % Compute next explicit piece

        strengths=gForces;
%         strengths = UpdateStrengthHex(sources,resting_length,vertical_length,diagonal_length); % hexagon
        
%         s_size=size(sources,1);
% %         strength1 = UpdateStrengthHex(sources(1:s_size/3,:),resting_length,vertical_length,diagonal_length); % hexagon
% %         strength2 = UpdateStrengthHex(sources(s_size/3+1:s_size*2/3,:),resting_length,vertical_length,diagonal_length);
% %         strength3 = UpdateStrengthHex(sources(s_size*2/3+1:s_size,:),resting_length,vertical_length,diagonal_length);
% %         strengths = [strength1;strength2;strength3];
%         strength1 = UpdateStrengthHex(sources(1:s_size/2,:),resting_length,vertical_length,diagonal_length);
%         strength2 = UpdateStrengthHex(sources(s_size/2+1:s_size,:),resting_length,vertical_length,diagonal_length);
%         strengths = [strength1;strength2];
        
        % convert data structure:(1,3*num)->(num,3)
        sources=reshape(xnew,num,3);
        
        % compute velocity (sources include bottom three stokeslets)
%         V=StokesletCUDA(sources,sources,strengths,deltas);
%       with GMRES, V is supposed to be 0.
        V=zeros(size(sources));
        
        % convert data structure back:(num,3)->(1,3*num)
        fenew=V(:)';
        
        % add background flow (x direction) to the first half of iterations
%         if k0 <= nbigsteps/2
%             fenew(1:num)=fenew(1:num)+sources(1:num,3)'*0.1;
%         end
       
        x(k0,:,mk+1) = xnew;    % Store solution
        fe(:,mk+1)=fenew;       % Store explicit function value
    end                       % end predictor
    
    %  Corrections loop
    for sdck = 2:niter                        %  Loop on SDC iterations
        
        tsdc = (k0-1)*bigDt;                    %  Reset time
        Immf = compintegral(sigmate,fe,sigmati,fi);     % Compute Residuals
        
        for mk = 1:nsdc-1                               %  time step
            %  correction
            % fi seems always zero(?) -ksl100118
            xnew = squeeze(x(k0,:,mk)) + (dtsdc(mk)*(fdiff - fi(:,mk+1))+Immf(:,mk))';
            tsdc = tsdc+dtsdc(mk);
            fdiff = squeeze(fe(:,mk+1));
            
            % original simple evaluation
            %[fenew] = compfe(xnew,tsdc);
            
            strengths=gForces;
%              strengths = UpdateStrengthHex(sources,resting_length,vertical_length,diagonal_length); % hexagon

%             s_size=size(sources,1);
% %             strength1 = UpdateStrengthHex(sources(1:s_size/3,:),resting_length,vertical_length,diagonal_length); % hexagon
% %             strength2 = UpdateStrengthHex(sources(s_size/3+1:s_size*2/3,:),resting_length,vertical_length,diagonal_length);
% %             strength3 = UpdateStrengthHex(sources(s_size*2/3+1:s_size,:),resting_length,vertical_length,diagonal_length);
% %             strengths = [strength1;strength2;strength3];
%             strength1 = UpdateStrengthHex(sources(1:s_size/2,:),resting_length,vertical_length,diagonal_length);
%             strength2 = UpdateStrengthHex(sources(s_size/2+1:s_size,:),resting_length,vertical_length,diagonal_length);
%             strengths = [strength1;strength2];
            
            % convert data structure:(1,3*num)->(num,3)
            sources=reshape(xnew,num,3);
            
            % compute velocity (sources include bottom three stokeslets)
%             V=StokesletCUDA(sources,sources,strengths,deltas);
%       with GMRES, V is supposed to be 0.
        V=zeros(size(sources));
        
            % convert data structure back:(num,3)->(1,3*num)
            fenew=V(:)';

            % add background flow (x direction) to the first half of iterations
%             if k0 < nbigsteps/2
%                 fenew(1:num)=fenew(1:num)+sources(1:num,3)'*0.1;
%             end
            
            x(k0,:,mk+1) = xnew;
            fe(:,mk+1)=fenew;
            
            fdiff =fenew'-fdiff;
        end           %  end correction sweep
    end             % end SDC  iterations

    %  Make a plot of solution so far
    plotme(xnew,dtsdc,nbigsteps,fenew,strengths,tracing,k0,deltas)  
    

%% 
% update plane tracing particles
    
%%
% update tracing particle     
    
    % visualize flow bubble
    sources=reshape(xnew,num,3);
    target=[tracing;sources];

    oldtracing = tracing;
    
     V=StokesletCUDA(target,sources,strengths,deltas);
     
%     if k0 <= nbigsteps/2
        for i=1:size(tracing,1)
            V(i,1)=V(i,1)+tracing(i,3)*0.1;
        end
%     end
    V = V(1:size(tracing,1),:);
    % update pos of flow bubble
    tracing=tracing+V*bigDt;
    
    draw_tracing_particles(oldtracing,tracing,'r*');

%     if(k0 > 1)
%         x1=xold;
%         x2=xnew;
%         xsize=size(x1,2)/3; % xxx yyy zzz
%         cnt=7;
%         oldtower=[x1(1:cnt:xsize)' x1(xsize+1:cnt:2*xsize)' x1(2*xsize+1:cnt:3*xsize)' ];
%         tower=[x2(1:cnt:xsize)' x2(xsize+1:cnt:2*xsize)' x2(2*xsize+1:cnt:3*xsize)' ];
%         draw_tracing_particles(oldtower,tower,'r*');
%         xold=xnew;
%     else
%         xold=xnew;
%     end
    

    %  pause
    toc
end
xend = xnew;   % Return the final solution

function draw_tracing_particles(oldtracing,tracing,color)
figure(3);hold on;grid on;
% axes('CameraPositionMode','manual','CameraViewAngleMode','manual','CameraTargetMode','manual');
for i=1:size(tracing,1)
    x=[oldtracing(i,1) tracing(i,1)];
    y=[oldtracing(i,2) tracing(i,2)];
    z=[oldtracing(i,3) tracing(i,3)];
    line(x,y,z);
end
% caxis=get(gcf,'CurrentAxes');campos=get(caxis,'CameraPosition');
plot3(tracing(:,1),tracing(:,2),tracing(:,3),color,'MarkerSize',5);
% set(caxis,'CameraPosition',campos)

function [V]=StokesletCUDA(target,sources,strengths,deltas);
% remove bottom three stokeslets from evaluation (seems no need. -ksl100117)
%             sources_nobottom = sources(4:num, :);
%             strengths_nobottom = strengths(4:num, :);
%             V_nobottom=StokesletEval(sources_nobottom,sources_nobottom,strengths_nobottom,deltas);
%             V=[zeros(3,3);V_nobottom];

% note currenlty only support constant delta.-ksl100312
% CUDA (stokeslet + image)
V=StokesletEval_CUDA(target,sources,strengths,deltas);
V=V+ImageEval_CUDA(target,sources,strengths,deltas);

% test overhead
% V=target*0;

% CPU
% V=StokesletEval(target,sources,strengths,deltas); 
% V=V+ImageEval(target,sources,strengths,deltas);

%------------------------------
function [Immf]=compintegral(sigmate,fe,sigmati,fi);
%------------------------------
%  Compute the integral term for the correction equation using the
%  integration matrices for the explicit and implicit parts
[nvals,nsdc]=size(fe);
Immf = zeros(nvals,nsdc-1);

%  Matrix vector multiply
for ik = 1:nsdc-1
    for jk = 1:nsdc
        Immf(:,ik)=Immf(:,ik)+sigmate(ik,jk)*fe(:,jk)+sigmati(ik,jk)*fi(:,jk);
    end
end

%------------------------------
function [fe]=compfe(xvect,t)
%------------------------------
%
%  Compute the explicit function evaluation
%
%  Here it is simply rotation plus vericle motion

npart=length(xvect)/3;
x = xvect(1:npart);
y = xvect(npart+1:2*npart);
z = xvect(2*npart+1:end);
r = sqrt((x.^2+y.^2));
theta = atan2(y,x);

fe = zeros(size(xvect));
fe(1:npart)=-2*pi*sin(theta).*r;
fe(npart+1:2*npart)=2*pi*cos(theta).*r;
fe(2*npart+1:end)=0.1*z;

%------------------------------
function [fe]=plotme(xvect,dtsdc,nbig,vvect,strengths,tracing,iteration,deltas)
%------------------------------
%
%  Plot the composite function in time
bigDt = sum(dtsdc);
bigt = [0:nbig-1]*bigDt;
npart=length(xvect)/3;
%position
x = xvect(1:npart);
y = xvect(npart+1:2*npart);
z = xvect(2*npart+1:end);
%velocity
vx = vvect(1:npart);
vy = vvect(npart+1:2*npart);
vz = vvect(2*npart+1:end);


strengths = strengths(:);


ax = strengths(1:npart);
ay = strengths(npart+1:2*npart);
az = strengths(2*npart+1:end);

figure(1);
% fixed current camera position
caxis=get(gcf,'CurrentAxes');campos=get(caxis,'CameraPosition');
% axes('CameraPositionMode','manual','CameraViewAngleMode','manual','CameraTargetMode','manual');
hold off;
quiver3(x,y,z,vx,vy,vz)
set(caxis,'CameraPosition',campos)
hold on;
plot3(x,y,z,'*');
% axis equal;
% visualize tracing particles
tx=tracing(:,1);
ty=tracing(:,2);
tz=tracing(:,3);
plot3(tx,ty,tz,'ro','MarkerSize',5);

title(sprintf('position & velocity, k=%d, d=%3.1f',iteration,deltas));
% axis([-1 2 -1 1 0 10]); % for hex

%axis equal;
% title(['position & velocity, k=',num2str(iteration)]);


% temporarily marked
% figure(2);
% % fixed current camera position
% caxis=get(gcf,'CurrentAxes');campos=get(caxis,'CameraPosition');
% % axes('CameraPositionMode','manual','CameraViewAngleMode','manual','CameraTargetMode','manual');
% %grid on
% hold off;
% quiver3(x,y,z,ax,ay,az)
% set(caxis,'CameraPosition',campos)
% hold on;
% plot3(x,y,z,'*');
% %view(-35,15)
% %view([0 0 1])
% %axis equal, axis([-1 6 -1 2 0 10 0 10]);
% title(['strength, k=',num2str(iteration)]);
% % axis([-2 8 -2 4 0 20]);
% axis([-0.1 0.15 -0.15 0.15 -0.1 1]);
% 



%pause(0.01);

function compute_velocity_grid(filename, sources, grid_size, x1,x2,y1,y2,z1,z2)
global gForces;
global deltas;

% plane_size=32;
% plane_y_tmp = linspace(  -2, 10,plane_size);
% plane_x_tmp = linspace(-1.6,1.6,plane_size);
% plane_z_tmp = linspace(   0, 10,plane_size);
plane_size=grid_size;
plane_y_tmp = linspace(y1,y2,plane_size);
plane_x_tmp = linspace(x1,x2,plane_size);
plane_z_tmp = linspace(z1,z2,plane_size);

final_V=[];
for k=1:plane_size
    fprintf('compute slice %d/%d...\n', k,plane_size);
    z=plane_z_tmp(k);
    plane_z = z * ones(plane_size * plane_size,1);
    plane_x = zeros(plane_size * plane_size,1);
    plane_y = zeros(plane_size * plane_size,1);
    for j=1:plane_size
        for i=1:plane_size
            plane_x((j-1)*plane_size + i) = plane_x_tmp(i);
            plane_y((j-1)*plane_size + i) = plane_y_tmp(j);
        end
    end
    plane_tracing=[plane_x plane_y plane_z];

    % sources=reshape(xnew,num,3);
    target=[plane_tracing;sources];
    strengths=gForces;
    V=StokesletCUDA(target,sources,strengths,deltas);

    size_plane_tracing=size(plane_tracing,1);
    for i=1:size(plane_tracing,1)
        V(i,1)=V(i,1)+plane_tracing(i,3)*0.1;
    end
    
    tmpV = V(1:size(plane_tracing,1),:);

% to visualize each xy slice
% V = V(1:size(plane_tracing,1),:);
% figure(4);
% quiver3(plane_x, plane_y, plane_z, V(:,1), V(:,2), V(:,3));
    
    final_V=[final_V;tmpV];
end

Vt=final_V';
fid=fopen(filename,'wb','n');
fwrite(fid,Vt,'float32');
fclose(fid);
