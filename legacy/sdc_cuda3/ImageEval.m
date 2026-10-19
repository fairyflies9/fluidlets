function [V] = ImageEval(targets,sources,strengths,deltas)
%
%  Evaluate the image of Stokeslet strengths to set 
%  a zero boundary condition in z
%
%  On input:
%    targets:    The location of the targests [Ntargets:3]
%    sources:    The location of the sources [Nsources:3]%    
%    strengths:  The vector of strengths [Nsources:3]    
%    deltas:     The size of the blobs.  Can be either size
%                [Nsources] or a scalar
%  On output:  
%    V :         Velocity at targets  [Ntargets:3]

%
   
[Ntargets,ndim]=size(targets);
if (ndim ~= 3)
  display(['bad dim in StokesletEval, ndim = ',num2str(ndim)]);
  pause;
end

[Nsources,ndim]=size(sources);
if (ndim ~= 3)
  display(['bad dim in StokesletEval, ndim = ',num2str(ndim)]);
  pause;
end

if ( size(sources) ~= size(strengths) )
    display('size mismatch StokesletEval');
    size(sources) 
    size(strengths)
    pause;
end

%  First make the sources the image
h0 =sources(:,3);    % Height of sources
sources(:,3)= -h0;   % Image of sources

%  Make blobs
  if length(deltas) == 1 
    delta = deltas;
    deltas = ones(1,Nsources)*delta;
  end
  d2 = deltas.^2;
  
%  Now loop over target points
  V = zeros(Ntargets,3);
  for i = 1:Nsources

    dx1 = targets(:,1)-sources(i,1);
    dx2 = targets(:,2)-sources(i,2);
    dx3 = targets(:,3)-sources(i,3);
    r2  = dx1.^2 + dx2.^2 + dx3.^2;

    % Stokeslets stuff
    H1 = 1/2 ./(r2+d2(i)).^(1/2) + d2(i)/2 ./(r2+d2(i)).^(3/2);
    H2 = 1/2 ./(r2+d2(i)).^(3/2);

    % Dipole stuff
    Hdp1 =  -h0(i)^2*( 1./(r2+d2(i)).^(3/2) - 3*d2(i)./(r2+d2(i)).^(5/2) );
    Hdp2 = -h0(i)^2*( -3./(r2+d2(i)).^(5/2) );
   
    %  Doublet stuff
    Hdb1 = 2*h0(i)*( -3/2*d2(i)./(r2+d2(i)).^(5/2) - 1/2 ./(r2+d2(i)).^(3/2) );
    Hdb2 = 2*h0(i)*( 1/2./(r2+d2(i)).^(3/2) );
    Hdb3 = 2*h0(i)*( -3/2./(r2+d2(i)).^(5/2) );

    %  Rotlet stuff
    Hr1 = (h0(i))*3*d2(i) ./(r2+d2(i)).^(5/2);
    
    %  Loop over targets
    for j = 1:Ntargets
      % A neg Stokeslet
      A(1,1) = -(H1(j) + dx1(j).*dx1(j).*H2(j));
      A(1,2) = -(dx1(j).*dx2(j).*H2(j)) ;
      A(1,3) = -(dx1(j).*dx3(j).*H2(j)) ;
      
      A(2,1) = -(dx2(j).*dx1(j).*H2(j)) ;
      A(2,2) = -(H1(j) + dx2(j).*dx2(j).*H2(j)) ;
      A(2,3) = -(dx2(j).*dx3(j).*H2(j)) ;
      
      A(3,1) = -(dx3(j).*dx1(j).*H2(j)) ;
      A(3,2) = -(dx3(j).*dx2(j).*H2(j)) ;
      A(3,3) = -(H1(j) + dx3(j).*dx3(j).*H2(j)) ;


      % A pos Dipole
      A(1,1) = A(1,1)-( Hdp1(j) + dx1(j).*dx1(j).*Hdp2(j) );  %i=1 j=1
      A(1,2) = A(1,2)-( dx1(j).*dx2(j).*Hdp2(j) );            %i=1 j=2
      A(1,3) = A(1,3)+( dx1(j).*dx3(j).*Hdp2(j) );            %i=1 j=3
      
      A(2,1) = A(2,1)-( dx2(j).*dx1(j).*Hdp2(j) );           %i=2 j=1
      A(2,2) = A(2,2)-( Hdp1(j) + dx2(j).*dx2(j).*Hdp2(j) ); %i=2 j=2
      A(2,3) = A(2,3)+ ( dx2(j).*dx3(j).*Hdp2(j) );          %i=2 j=3
      
      A(3,1) = A(3,1)-( dx3(j).*dx1(j).*Hdp2(j) );           %i=3 j=1
      A(3,2) = A(3,2)-( dx3(j).*dx2(j).*Hdp2(j) );           %i=3 j=2
      A(3,3) = A(3,3)+( Hdp1(j) + dx3(j).*dx3(j).*Hdp2(j) ); %i=3 j=3

      % A pos Doublet
      A(1,1) = A(1,1)-( dx3(j).*(Hdb2(j) + dx1(j).*dx1(j).*Hdb3(j)) ) ;
      A(1,2) = A(1,2)-( dx1(j).*dx2(j).*dx3(j).*Hdb3(j) );     % i=1 j=2
      A(1,3) = A(1,3)+dx1(j).*(Hdb2(j) + dx3(j).*dx3(j).*Hdb3(j) );   % i=1 j=3
      
      A(2,1) = A(2,1)-( dx2(j).*dx1(j).*dx3(j).*Hdb3(j) ) ;           % i=2 j=1
      A(2,2) = A(2,2)-( dx3(j).*(Hdb2(j) + dx2(j).*dx2(j).*Hdb3(j)) ) ; 
      A(2,3) = A(2,3)+dx2(j).*(Hdb2(j)+ dx3(j).*dx3(j).*Hdb3(j) ) ; % i=2 j=3
      
      A(3,1) = A(3,1)-( dx1(j).*(Hdb1(j) + dx3(j).*dx3(j).*Hdb3(j) ) ); 
      A(3,2) = A(3,2)-( dx2(j).*(Hdb1(j) + dx3(j).*dx3(j).*Hdb3(j) ) );
      A(3,3) = A(3,3)+ dx3(j).*(Hdb1(j) + 2*Hdb2(j) + dx3(j).*dx3(j).*Hdb3(j));
      
      % A pos Rotlet
      A(1,1) = A(1,1) + dx3(j).*Hr1(j);
      A(2,2) = A(2,2) + dx3(j).*Hr1(j);
      A(3,1) = A(3,1)-dx1(j).*Hr1(j);
      A(3,2) = A(3,2)-dx2(j).*Hr1(j);
      
      V(j,:) = V(j,:) + (A*strengths(i,:)')'/(4*pi);    
    end
  end