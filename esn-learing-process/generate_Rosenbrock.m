function [ x1,x2,y ] = generate_Rosenbrock( xmin,xmax,points )
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
x1=linspace(xmin,xmax,points);
%x1=2*(xmax-xmin)*rand(points)-xmax;
x2=linspace(xmin,xmax,points);
%x2=2*(xmax-xmin)*rand(points)-xmax;
[x1,x2]=meshgrid(x1,x2);
f1=100*[(x1.^2-x2)].^2;
f2=(1-x1).^2;
y= 1+f1-f2;

end

