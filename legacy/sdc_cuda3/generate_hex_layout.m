function [ total_data ] = generate_hex_layout( delta, height )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% test data
total_data=[0 0 0 0 10 20;
0 0 4 0 10 20;
0 0 8 0 10 20;
  ];
return;

% according to the tower layout from Laura.
dy = delta;
dx = delta*sqrt(3)/2;

total_data=[];
total_data=[total_data;add_one_line(    0,  2*dy,dy,2,height)];

% it costs about 3 more min to compute gmres with following 2 lines
% total_data=[total_data;add_one_line(   dx,0.5*dy,dy,5,height)];
% total_data=[total_data;add_one_line( 2*dx,     0,dy,6,height)];

% note GPU in Alex's laptop would be out of memory if adding all following lines.
% total_data=[total_data;add_one_line( 3*dx,0.5*dy,dy,5,height)];
% total_data=[total_data;add_one_line( 4*dx,     0,dy,6,height)];
% total_data=[total_data;add_one_line( 5*dx,0.5*dy,dy,5,height)];
% total_data=[total_data;add_one_line( 6*dx,     0,dy,6,height)];
% total_data=[total_data;add_one_line( 7*dx,0.5*dy,dy,5,height)];
% total_data=[total_data;add_one_line( 8*dx,     0,dy,6,height)];
% total_data=[total_data;add_one_line( 9*dx,0.5*dy,dy,5,height)];
% total_data=[total_data;add_one_line(10*dx,     0,dy,6,height)];
% total_data=[total_data;add_one_line(11*dx,0.5*dy,dy,5,height)];
% total_data=[total_data;add_one_line(12*dx,  2*dy,dy,2,height)];
    
end

function[ positions ] = add_one_line(x, yoffset, yspacing, number, height)
    positions=[];
    for i=0:number-1
        y = yoffset + yspacing*i;
        positions = [positions; 0 x y 0 height 20];     % level=20
    end
end

