function [ x0 ] = build_line_tower(origin)
%UNTITLED Summary of this function goes here
%   origin: [ x y z ], 1 by 3 vector

%distance of stokeslet
% level_num, radius are based on the paper's configuration
level_num = 20;
radius=0.5;
total_height=10;
% radius=0.037; % the same, no move.
% total_height=1;
level_height=total_height/level_num;
k=1;
% x part
for i=0:level_num
    x0(k) = origin(1);
    k=k+1;
end
% y part
for i=0:level_num
    x0(k) = origin(2);
    k=k+1;
end
% z part
for i=0:level_num
    height=i*level_height;
    x0(k) = origin(3) + height;
    k=k+1;
end


end

