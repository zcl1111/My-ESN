function y = MG( t,x,z )
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
xlag=z(1,:);
y=ones(1,1);
y(1)=(0.2*xlag(1))/(1+(xlag(1))^10)-0.1*x(1);


end

