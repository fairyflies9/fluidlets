%rand seed
rand('state',124);
% for n=1:50
fprintf('compare results from Matlab and Cuda\n');
% TODO: currently, target size has to be larger than source's.	-ksl100412
% for n=45:52
for n=30:40
sources=rand(n,3);
targets=rand(3*n,3);
% targets=rand(10,3);

% targets=sources;
%targets=ones(n,3)
%targets(1,:)=[2,3,5]
% strength=ones(n,3);
strength=rand(n,3);
deltas=0.8;

% load save_strength;
% load save_tracing;
% load save_source;
% targets=tracing;
% strength=strengths;

% tic;
% fprintf('n=%d, MATLAB version...\n',size)
% V_MAT=StokesletEval(targets,sources,strength,deltas);
V_MAT=ImageEval(targets,sources,strength,deltas);
% toc;

% tic;
% fprintf('\nn=%d, CUDA version...\n',size)
% V_CUDA=StokesletEval_CUDA(targets,sources,strength,deltas);
V_CUDA=ImageEval_CUDA(targets,sources,strength,deltas);
% toc;

CUDA=V_CUDA(1,1);
MATLAB=V_MAT(1,1);

%V_MAT = ones(size, 3)
%V_CUDA = V_MAT + 0.5

% relative percentage
diff=(V_CUDA-V_MAT) ./ V_MAT;
% [n:1]
err_sqrt = sqrt(diff(:,1).^2 + diff(:,2).^2 + diff(:,3).^2);
% relative error
error=mean(err_sqrt);
fprintf('n=%d, CUDA:%f, Mat:%f, err=%f\n',n,CUDA,MATLAB,error);
% display(['n=' num2str(n) ', CUDA:' num2str(CUDA) ', Mat:' num2str(MATLAB) ', err=' num2str(error) ])
%relerr = sum(transpose((diff./V_MAT).^2)).^0.5
%V_CUDA
end