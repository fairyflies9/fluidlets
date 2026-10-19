function [ x0 ] = build_tri_tower(origin)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%distance of stokeslet
% level_num, radius are based on the paper's configuration
level_num = 10;
radius=0.037;
level_height=1/level_num
k=1;
% x part
for i=1:level_num+1
    x0(k)=-radius + origin(1);
    x0(k+1)=radius*cos(pi/3) + origin(1);
    x0(k+2)=radius*cos(pi/3) + origin(1);
    k=k+3;
end
% y part
for i=1:level_num+1
    x0(k)=0 + origin(2);
    x0(k+1)=-radius*sin(pi/3) + origin(2);
    x0(k+2)=radius*sin(pi/3) + origin(2);
    k=k+3;
end
% z part
for i=1:level_num+1
    height=(i-1)*level_height;
    x0(k)=height + origin(3);
    x0(k+1)=height + origin(3);
    x0(k+2)=height + origin(3);
    k=k+3;
end

end

