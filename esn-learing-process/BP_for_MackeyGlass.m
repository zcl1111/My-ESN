clc
clear
data = load('MackeyGlass_t17.txt');
traininput=data(1:2000)';
trainoutput=data(1:2000)';
testinput=data(2001:4000)';
testoutput=data(2001:4000)';
%[p1,minp,maxp,t1,mint,maxt]=premnmx(traininput,trainoutput);

%运行50次求平均值
for i=1:50
net=newff(minmax(traininput),[1,20,1],{'tansig','tansig','purelin'},'trainlm');
% 学习速率为0.05。
net.trainParam.lr=0.05;
%设置训练次数为100次，运行8秒
net.trainParam.epochs = 100;
%设置收敛误差为0
%net.trainParam.goal=0.0000001;
%训练网络
[net,tr]=train(net,traininput,trainoutput);
%训练输出值
y1=sim(net,traininput);
%测试输出值
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




%单独运行一次的程序
if 0
net=newff(minmax(traininput),[1,20,1],{'tansig','tansig','purelin'},'trainlm');
% 学习速率为0.05。
net.trainParam.lr=0.05;
%设置训练次数
net.trainParam.epochs = 100;
%设置收敛误差
%net.trainParam.goal=0.0000001;
%训练网络
[net,tr]=train(net,traininput,trainoutput);
%训练输出值
y1=sim(net,traininput);
%测试输出值
y2=sim(net,testinput);

figure;
plot(testinput,'r-');
hold on;
plot(y2,'b.');
hold off;   
%计算训练误差
[NRMSE,Absolute_err,less1,less2,more2] = compute_error(y1, trainoutput);
disp(sprintf('train NRMSE = %s', num2str(NRMSE)));
disp(sprintf('train Absolute_err = %s', num2str(Absolute_err)));
disp(sprintf('train less1 = %s', num2str(less1)));
disp(sprintf('train less2 = %s', num2str(less2)));
disp(sprintf('train more2 = %s', num2str(more2)));
%计算测试误差
[NRMSE,Absolute_err,less1,less2,more2] = compute_error(y2,testoutput); 
disp(sprintf('test NRMSE = %s', num2str(NRMSE)));
disp(sprintf('test Absolute_err = %s', num2str(Absolute_err)));
disp(sprintf('test less1 = %s', num2str(less1)));
disp(sprintf('test less2 = %s', num2str(less2)));
disp(sprintf('test more2 = %s', num2str(more2)));
end
 

 