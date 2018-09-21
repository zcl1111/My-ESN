clear
%这个程序调好了，用的是GriewangK函数，x取值范围为-2.048~2.048，z范围为0.0013~1.4601
%进行比较神经元个数对网络的影响
trainLen = 2000; 
testLen = 500; 
initLen = 100; 
inSize = 2; %输入维度K
outSize = 1;
resSize = 1200
for i=1:10
data=load('rosebrock_rand2500.txt');
x=data(:,[1:2]);
y=data(:,3);

% generate the ESN reservoir 

%输入输出归一化处理
[x0,ps1]=mapminmax(x',0,1);
x0=x0';
[y0,ps2]=mapminmax(y',0,1);
y0=y0';
trainInputSequence=x0((1:trainLen),:);
trainOutputSequence=y0((1:trainLen),:);

testInputSequence=x0((trainLen+1:trainLen+testLen),:);
testOutputSequence=y0((trainLen+1:trainLen+testLen),:);


esn = generate_esn4(inSize, resSize,outSize, ...
   'spectralRadius',0.001,'inputScaling',[1;1],'inputShift',[0;0], ...
    'teacherScaling',[1],'teacherShift',[0],'feedbackScaling', 0, ...
    'type', 'plain_esn'); 

esn.internalWeights = esn.spectralRadius * esn.internalWeights_UnitSR;%谱半径为0.5
nForgetPoints = initLen;
[trainedEsn ,stateMatrix] = ...
    train_esn(trainInputSequence, trainOutputSequence, esn, nForgetPoints) ;

%%%% plot the internal states of 4 units
nPoints = 100 ; 
plot_states(stateMatrix,[1 2 3 4], nPoints, 1) ; 

% compute the output of the trained ESN on the training and testing data,
% discarding the first nForgetPoints of each
nForgetPoints =20 ; 
predictedTrainOutput = test_esn(trainInputSequence, trainedEsn, nForgetPoints);
%训练预测输出反归一化
t1=mapminmax('reverse',predictedTrainOutput',ps2);
predictedTrainOutput=t1';
trainOutputSequence=y((1:trainLen),:);

predictedTestOutput = test_esn(testInputSequence,  trainedEsn, nForgetPoints) ; 
%测试输出反归一化
t2=mapminmax('reverse',predictedTestOutput',ps2);
predictedTestOutput=t2';
testOutputSequence=y((trainLen+1:trainLen+testLen),:);
% create input-output plots
figure(11);
%nPlotPoints = 100 ; 
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('测试教师数据 (红色实线) vs 预测数据 (蓝色实心点)');
xlim([0,200]);

%%%%compute training error

%%%%计算训练误差
[NRMSE1,Absolute_err1,less11,less21,more21] = compute_error3(predictedTrainOutput, trainOutputSequence);
a1(i)=NRMSE1;
b1(i)=Absolute_err1;
c1(i)=less11;
d1(i)=less21;
e1(i)=more21;

[NRMSE2,Absolute_err2,less12,less22,more22] = compute_error3(predictedTestOutput,testOutputSequence);
a2(i)=NRMSE2;
b2(i)=Absolute_err2;
c2(i)=less12;
d2(i)=less22;
e2(i)=more22;
end
a1=sum(a1)/10;
b1=sum(b1)/10;
c1=sum(c1)/10;
d1=sum(d1)/10;
e1=sum(e1)/10;
a2=sum(a2)/10;
b2=sum(b2)/10;
c2=sum(c2)/10;
d2=sum(d2)/10;
e2=sum(e2)/10;
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


