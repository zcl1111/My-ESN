%clc;
%clear;
traininput1=load('traininputsequence1.txt');
traininput=traininput1';
trainoutput1=load('traininputsequence1.txt');
trainoutput=trainoutput1';
testinput1=load('testinputsequence1.txt');
testinput=testinput1';
testoutput1=load('testinputsequence1.txt');
testoutput=testoutput1';


%����50����ƽ��ֵ
if 0
for i=1:50
net=newff(minmax(traininput),[1,20,1],{'tansig','tansig','purelin'},'trainlm');
% ѧϰ����Ϊ0.05��
net.trainParam.lr=0.05;
%����ѵ������Ϊ100�Σ�����8��
net.trainParam.epochs = 100;
%�����������Ϊ0
%net.trainParam.goal=0.0000001;
%ѵ������
[net,tr]=train(net,traininput,trainoutput);
%ѵ�����ֵ
y1=sim(net,traininput);
%�������ֵ
y2=sim(net,testinput);

[NRMSE1,Absolute_err1,less11,less21,more21] = compute_error(y1, trainoutput);

a1(i)=NRMSE1;
b1(i)=Absolute_err1;
c1(i)=less11;
d1(i)=less21;
e1(i)=more21;

[NRMSE2,Absolute_err2,less12,less22,more22] = compute_error(y2,testoutput);
a2(i)=NRMSE2;
b2(i)=Absolute_err2;
c2(i)=less12;
d2(i)=less22;
e2(i)=more22;
end
a1=sum(a1)/50;
b1=sum(b1)/50;
c1=sum(c1)/50;
d1=sum(d1)/50;
e1=sum(e1)/50;
a2=sum(a2)/50;
b2=sum(b2)/50;
c2=sum(c2)/50;
d2=sum(d2)/50;
e2=sum(e2)/50;
disp(sprintf('train NRMSE = %s', num2str(a1)));
disp(sprintf('train Absolute_err = %s', num2str(b1)));
disp(sprintf('train less1 = %s', num2str(c1)));
disp(sprintf('train less2 = %s', num2str(d1)));
disp(sprintf('train more2 = %s', num2str(e1)));

disp(sprintf('test NRMSE = %s', num2str(a2)));
disp(sprintf('test Absolute_err = %s', num2str(b2)));
disp(sprintf('test less1 = %s', num2str(c2)));
disp(sprintf('test less2 = %s', num2str(d2)));
disp(sprintf('test more2 = %s', num2str(e2)));
end 


%��������һ�εĳ���
%if 0
net=newff(minmax(traininput),[1,20,1],{'tansig','tansig','purelin'},'trainlm');
% ѧϰ����Ϊ0.05��
net.trainParam.lr=0.05;
%����ѵ������
net.trainParam.epochs = 500;
%�����������
%net.trainParam.goal=0.0000001;
%ѵ������
[net,tr]=train(net,traininput,trainoutput);
%ѵ�����ֵ
y1=sim(net,traininput);
%�������ֵ
y2=sim(net,testinput);

figure;
plot(testinput,'r-');
hold on;
plot(y2,'b.');
hold off;   
%����ѵ�����
[NRMSE,Absolute_err,less1,less2,more2] = compute_error(y1, trainoutput);
disp(sprintf('train NRMSE = %s', num2str(NRMSE)));
disp(sprintf('train Absolute_err = %s', num2str(Absolute_err)));
disp(sprintf('train less1 = %s', num2str(less1)));
disp(sprintf('train less2 = %s', num2str(less2)));
disp(sprintf('train more2 = %s', num2str(more2)));

%����������
[NRMSE,Absolute_err,less1,less2,more2] = compute_error(y2,testoutput); 
disp(sprintf('test NRMSE = %s', num2str(NRMSE)));
disp(sprintf('test Absolute_err = %s', num2str(Absolute_err)));
disp(sprintf('test less1 = %s', num2str(less1)));
disp(sprintf('test less2 = %s', num2str(less2)));
disp(sprintf('test more2 = %s', num2str(more2)));
%end
 

 