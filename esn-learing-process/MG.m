function y = MG( t,x,z )
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
xlag=z(1,:);
y=ones(1,1);
y(1)=(0.2*xlag(1))/(1+(xlag(1))^10)-0.1*x(1);


end

