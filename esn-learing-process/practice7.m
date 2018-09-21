clear
clc
[x1,x2,f] = generate_GriewangK(-5,5,50);
[ x,z ] =convert( x1,x2,f);
GriewangK=[x,z];
a=randperm(2500);
GriewangK=GriewangK(a,:);
figure(10); 
surf_sequence(x2,x1,f); 
title('2500组数据的三维图'); 