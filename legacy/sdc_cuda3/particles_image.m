function [xend]=particles_image(x0,Tfin,bigDt)
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
figure(1);
clf;
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
niter = 9;                    % Total number of iterations (order
                              % of method)
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
%fenew=compfe(x0,0);       % Compute initial explicit  function
%fenew=comp_stokeslets(x0,0);

% convert data structure
num = size(x0,2)/3
sources=zeros(num,3);
for i=1:num
    sources(i,1)=x0(i);
    sources(i,2)=x0(num + i);
    sources(i,3)=x0(num*2 + i);
end

strengths = zeros(num,3);
deltas = 1;
V=StokesletEval(sources,sources,strengths,deltas);

% convert data structure back
for i=1:num
    fenew(i)=V(i,1);
    fenew(i)=V(i,1)+sources(i,3)*0.1; % not squashed
    %fenew(i)=V(i,1)+sources(i,3)*0.12; % not squashed
    %fenew(i)=V(i,1)+sources(i,3)*0.13; % squashed after 3 iterations
    fenew(num + i)=V(i,2);
    fenew(num*2 + i)=V(i,3);
end

%still squashed!
%stokes in the first level don't move.
for i=1:3
    fenew(i)=0;
    fenew(num + i)=0;
    fenew(num*2 + i)=0;
end

fe(:,1)=squeeze(fenew);   % Store it 
x(1,:,1)=x0;              % Store initial condition

%-----------------------------------------------
%  Main time step loop
%----------------------------------------------
for k0=1:nbigsteps               %  Time step loop
  tsdc = (k0-1)*bigDt;           %  Time at beginning of time step
  if(k0 > 1)                     
    x(k0,:,1)=x(k0-1,:,end);     %  Resetting solution 
    fe(:,1)=fe(:,end);           %  Resetting explicit function
  end

  %  Predictor loop
  for mk = 1:nsdc-1                                   % begin predictor loop 
    xnew = squeeze(x(k0,:,mk)) + dtsdc(mk)*fe(:,mk)';  %  Forward Euler step
    tsdc = tsdc+dtsdc(mk);                             %  Update time
    %[fenew] = compfe(xnew,tsdc);       % Compute next explicit piece
    %[fenew] = comp_stokeslets(xnew,tsdc);
 %------------
        for i=1:num
            sources(i,1)=xnew(i);
            sources(i,2)=xnew(num + i);
            sources(i,3)=xnew(num*2 + i);
        end

        strengths = UpdateStrength(sources);
        deltas = 1;
        V=StokesletEval(sources,sources,strengths,deltas);

        % convert data structure back
        for i=1:num
            fenew(i)=V(i,1)+sources(i,3)*0.1;
            %fenew(i)=V(i,1);
            fenew(num + i)=V(i,2);
            fenew(num*2 + i)=V(i,3);
        end
        
        % stokes in the first level don't move.
        for i=1:3
            fenew(i)=0;
            fenew(num + i)=0;
            fenew(num*2 + i)=0;
        end
        
%-----------    
    
    x(k0,:,mk+1) = xnew;                      % Store solution   
    fe(:,mk+1)=fenew;                         % Store explicit function value
  end                                                  % end predictor
						       
  %  Corrections loop
  for sdck = 2:niter                        %  Loop on SDC iterations
    tsdc = (k0-1)*bigDt;                    %  Reset time
    Immf = compintegral(sigmate,fe,sigmati,fi);     % Compute Residuals
    for mk = 1:nsdc-1                               %  time step
                                                    %  correction

      xnew = squeeze(x(k0,:,mk)) + (dtsdc(mk)*(fdiff - fi(:,mk+1))+Immf(:,mk))';
      tsdc = tsdc+dtsdc(mk);
      fdiff = squeeze(fe(:,mk+1));
      %[fenew] = compfe(xnew,tsdc);
      %[fenew] = comp_stokeslets(xnew,tsdc);
      
 %------------
        for i=1:num
            sources(i,1)=xnew(i);
            sources(i,2)=xnew(num + i);
            sources(i,3)=xnew(num*2 + i);
        end

        strengths = UpdateStrength(sources);
        deltas = 1;
        
        V=StokesletEval(sources,sources,strengths,deltas);

        % convert data structure back
        for i=1:num
            fenew(i)=V(i,1)+sources(i,3)*0.1;
            %fenew(i)=V(i,1);
            fenew(num + i)=V(i,2);
            fenew(num*2 + i)=V(i,3);
        end
        
        % stokes in the first level don't move.
        for i=1:3
            fenew(i)=0;
            fenew(num + i)=0;
            fenew(num*2 + i)=0;
        end
        
%-----------

      x(k0,:,mk+1) = xnew; fe(:,mk+1)=fenew; 
      fdiff =fenew'-fdiff;
    end                                           %  end correction sweep
  end                                     % end SDC  iterations
					  
  plotme(xnew,dtsdc,nbigsteps,fenew)  %  Make a plot of solution so far
  grid
%  pause
end
 
xend = xnew;   % Return the final solution
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
function [fe]=plotme(xvect,dtsdc,nbig,vvect)
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

hold off;
quiver3(x,y,z,vx,vy,vz)
hold on;
plot3(x,y,z,'*');





pause(0.01);

