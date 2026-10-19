function [V] = StokesSelf(strengths)
%
%  Evaluate the Stokeslet velocity at the targets from the 
%  strengths at the sources.  This version
%  is to be used with GMRES, hence needs to have
%  targets, sources, and deltas global
%
%  The length of targets and sources should be the same
%
%  On input:
%    targets:    The location of the targests [Ntargets:3]
%    sources:    The location of the sources [Nsources:3]%    
%    strengths:  The vector of strengths [Nsources:3]    
%    deltas:     The size of the blobs.  Can be either size
%                [Nsources] or a scalar
%  On output:  
%    V :         Velocity at targets  [Ntargets:3]

global sources
global targets
global deltas
global iter_count
iter_count=iter_count+1;

% fprintf('iter %d\n',iter_count);

if ( size(sources) ~= size(strengths) )
    display('size mismatch StokesletEval');
    size(sources) 
    size(strengths)
    pause;
end

Nvar = length(sources);
Nsources = Nvar/3;
Nvar = length(targets);
Ntargets = Nvar/3;

ndim=3;
Mtargets = reshape(targets,Ntargets,ndim);
Msources = reshape(sources,Nsources,ndim);
Mstrengths = reshape(strengths,Nsources,ndim);

if (ndim ~= 3)
  display(['bad dim in StokesletEval, ndim = ',num2str(ndim)]);
  pause;
end

[Nsources,ndim]=size(Msources);
if (ndim ~= 3)
  display(['bad dim in StokesletEval, ndim = ',num2str(ndim)]);
  pause;
end

% using GPU
tmpV=StokesletEval_CUDA(Mtargets,Msources,Mstrengths,deltas);
tmpV=tmpV+ImageEval_CUDA(Mtargets,Msources,Mstrengths,deltas);

% using CPU
% tmpV=StokesletEval(Mtargets,Msources,Mstrengths,deltas);
% tmpV=tmpV+ImageEval(Mtargets,Msources,Mstrengths,deltas);

V=reshape(tmpV,Nvar,1);
return;
  
%  Make blobs
  if length(deltas) == 1 
    delta = deltas;
    deltas = ones(1,Nsources)*delta;
  end
  d2 = deltas.^2;
  
%  Now loop over target points
  tmpV = zeros(Ntargets,3);
  for i = 1:Nsources

    dx1 = Mtargets(:,1)-Msources(i,1);
    dx2 = Mtargets(:,2)-Msources(i,2);
    dx3 = Mtargets(:,3)-Msources(i,3);
    r2  = dx1.^2 + dx2.^2 + dx3.^2;
    
    H1 = 1/2 ./(r2+d2(i)).^(1/2) + d2(i)/2 ./(r2+d2(i)).^(3/2);
    H2 = 1/2 ./(r2+d2(i)).^(3/2);

    %  Loop over targets

    for j = 1:Ntargets
      A(1,1) = H1(j) + dx1(j).*dx1(j).*H2(j);
      A(1,2) = dx1(j).*dx2(j).*H2(j) ;
      A(1,3) = dx1(j).*dx3(j).*H2(j) ;
      
      A(2,1) = dx2(j).*dx1(j).*H2(j) ;
      A(2,2) = H1(j) + dx2(j).*dx2(j).*H2(j) ;
      A(2,3) = dx2(j).*dx3(j).*H2(j) ;
      
      A(3,1) = dx3(j).*dx1(j).*H2(j) ;
      A(3,2) = dx3(j).*dx2(j).*H2(j) ;
      A(3,3) = H1(j) + dx3(j).*dx3(j).*H2(j) ;

%      V(j,:) = V(j,:) + (strengths(i,:)*A)'/(4*pi);    
   tmpV(j,:) = tmpV(j,:) + (A*Mstrengths(i,:)')'/(4*pi);    
    end
  end
  
  
  