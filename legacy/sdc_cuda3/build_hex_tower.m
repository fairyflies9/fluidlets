function [ x0 ] = build_hex_tower(origin)
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
    for j=0:5
        x0(k+j) = origin(1) + radius * cos(pi/6 + 2*pi / 6 * j);
    end
    x0(k+6) = origin(1) + 0;
    k=k+7;
end
% y part
for i=0:level_num
    for j=0:5
        x0(k+j) = origin(2) + radius * sin(pi/6 + 2*pi / 6 * j);
    end
    x0(k+6) = origin(2) + 0;
    k=k+7;
end
% z part
for i=0:level_num
    height=i*level_height;
    for j=0:6
        x0(k+j) = origin(3) + height;
    end
    k=k+7;
end



end

