clear;
clc
tau=17;
sol=dde23('MG',tau,0.92,[1,1000]);
figure;
plot(sol.x,sol.y,'r');
