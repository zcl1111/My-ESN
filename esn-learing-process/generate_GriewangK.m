function [ x1,x2,f ] =generate_GriewangK(xmin,xmax,points)
%UNTITLED2 �˴�x1,x2Ϊmeshgeid���������ʽ
%   �˴���ʾ��ϸ˵��
x1=linspace(xmin,xmax,points);
%x1=2*(xmax-xmin)*rand(points)-xmax;
x2=linspace(xmin,xmax,points);
%x2=2*(xmax-xmin)*rand(points)-xmax;
[x1,x2]=meshgrid(x1,x2);
f1=(x1.^2+x2.^2)/4000;
f2=cos(x1).*cos(x2/sqrt(2));
f= 1+f1-f2;
end

