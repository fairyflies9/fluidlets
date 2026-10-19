function [strength]=UpdateStrength(x, resting_length, resting_length_diagonal)
[totalSize, tmp]=size(x);
strength=zeros(totalSize,3);

% how many levels in the glycocalyx tower
iteration=totalSize/3 - 1;

%resting_length=1;
%resting_length_diagonal=norm([resting_length,resting_length]);
k=1;
% iterate from the bottom level to the top.
for i=1:iteration
    [strength] = update_pair(x,strength,k  ,k+1,resting_length);
    [strength] = update_pair(x,strength,k  ,k+2,resting_length);
    [strength] = update_pair(x,strength,k+1,k+2,resting_length);
    [strength] = update_pair(x,strength,k  ,k+3,resting_length);
    [strength] = update_pair(x,strength,k+1,k+4,resting_length);
    [strength] = update_pair(x,strength,k+2,k+5,resting_length);
    
    [strength] = update_pair(x,strength,k  ,k+4,resting_length_diagonal);
    [strength] = update_pair(x,strength,k+1,k+3,resting_length_diagonal);
    [strength] = update_pair(x,strength,k+1,k+5,resting_length_diagonal);
    [strength] = update_pair(x,strength,k+2,k+4,resting_length_diagonal);
    [strength] = update_pair(x,strength,k  ,k+5,resting_length_diagonal);
    [strength] = update_pair(x,strength,k+2,k+3,resting_length_diagonal);
    
    k=k+3;
end

% the top level
[strength] = update_pair(x,strength,k,k+1,resting_length);
[strength] = update_pair(x,strength,k,k+2,resting_length);
[strength] = update_pair(x,strength,k+1,k+2,resting_length);


function [strength_out] = update_pair(x, strength_in, idx_a, idx_b, resting_length)
% if >= 420, strange positions.
spring_constant=410; % seems normal
spring_constant=411; % seems normal
spring_constant=413; % seems normal
spring_constant=418; % 418 : abnormal after 3 iterations, ok with steps=60
spring_constant=300; % based on the paper's configuration
strength_out = strength_in;
%strength_pair = x(idx_a) - x(idx_b);

ab = x(idx_b,:) - x(idx_a,:);
%compute the "pair_strength" according to length of ab and the resting
%length
L = norm(ab);
norm_ab = ab / L;
% L > resting_length, pair_strength is positive (direction: a->b)
% L < resting_length, pair_strength is negative (direction: b->a)
pair_strength = norm_ab * spring_constant * (L - resting_length);

strength_out(idx_a,:) = strength_out(idx_a,:) + pair_strength;
strength_out(idx_b,:) = strength_out(idx_b,:) - pair_strength;

