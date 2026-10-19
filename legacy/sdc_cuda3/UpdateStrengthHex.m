function [strength]=UpdateStrengthHex(x,resting_length,vertical_length,diagonal_length)
[totalSize, tmp]=size(x);
strength=zeros(totalSize,3);

% 7 points per level for hexagon
num_point_per_level = 7;
% how many levels in the glycocalyx tower 
iteration=totalSize / num_point_per_level - 1;

%resting_length=1;
%resting_length_diagonal=norm([resting_length,resting_length]);
k=1;
% iterate from the bottom level to the top.
for i=1:iteration
    [strength] = update_pair(x,strength,k  ,k+6,resting_length);
    [strength] = update_pair(x,strength,k+1,k+6,resting_length);
    [strength] = update_pair(x,strength,k+2,k+6,resting_length);
    [strength] = update_pair(x,strength,k+3,k+6,resting_length);
    [strength] = update_pair(x,strength,k+4,k+6,resting_length);
    [strength] = update_pair(x,strength,k+5,k+6,resting_length);
    [strength] = update_pair(x,strength,k  ,k+1,resting_length);
    [strength] = update_pair(x,strength,k+1,k+2,resting_length);
    [strength] = update_pair(x,strength,k+2,k+3,resting_length);
    [strength] = update_pair(x,strength,k+3,k+4,resting_length);
    [strength] = update_pair(x,strength,k+4,k+5,resting_length);
    [strength] = update_pair(x,strength,k+5,k  ,resting_length);
    
%     vertical edges
    for j=0:6
        [strength] = update_pair(x,strength,k+j,k+j+7,vertical_length);
    end
    
%     diag edges
    [strength] = update_pair(x,strength,k  ,k+8,diagonal_length);
    [strength] = update_pair(x,strength,k+1,k+7,diagonal_length);
    
    [strength] = update_pair(x,strength,k+1,k+9,diagonal_length);
    [strength] = update_pair(x,strength,k+2,k+8,diagonal_length);
    
    [strength] = update_pair(x,strength,k+2,k+10,diagonal_length);
    [strength] = update_pair(x,strength,k+3,k+9,diagonal_length);
    
    [strength] = update_pair(x,strength,k+3,k+11,diagonal_length);
    [strength] = update_pair(x,strength,k+4,k+10,diagonal_length);
    
    [strength] = update_pair(x,strength,k+4,k+12,diagonal_length);
    [strength] = update_pair(x,strength,k+5,k+11,diagonal_length);

    [strength] = update_pair(x,strength,k+5,k+ 7,diagonal_length);
    [strength] = update_pair(x,strength,k  ,k+12,diagonal_length);

    k=k+7;
end

% the top level
[strength] = update_pair(x,strength,k  ,k+6,resting_length);
[strength] = update_pair(x,strength,k+1,k+6,resting_length);
[strength] = update_pair(x,strength,k+2,k+6,resting_length);
[strength] = update_pair(x,strength,k+3,k+6,resting_length);
[strength] = update_pair(x,strength,k+4,k+6,resting_length);
[strength] = update_pair(x,strength,k+5,k+6,resting_length);
[strength] = update_pair(x,strength,k  ,k+1,resting_length);
[strength] = update_pair(x,strength,k+1,k+2,resting_length);
[strength] = update_pair(x,strength,k+2,k+3,resting_length);
[strength] = update_pair(x,strength,k+3,k+4,resting_length);
[strength] = update_pair(x,strength,k+4,k+5,resting_length);
[strength] = update_pair(x,strength,k+5,k  ,resting_length);

function [strength_out] = update_pair(x, strength_in, idx_a, idx_b, resting_length)
spring_constant=120; 
spring_constant=150; % unstable with 120 steps
spring_constant=300; % stable with 240 steps for hexagon
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

