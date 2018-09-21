function [ x,z ] =convert( x1,x2,f)
%UNTITLED7 把三维x1,x2,z矩阵向量变为一一对应的二维X-Z数据向量
%   此处显示详细说明
x1=x1(:);
x2=x2(:);
x=[x1,x2];
z=f(:);

end

