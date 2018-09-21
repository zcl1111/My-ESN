clear
trainLen = 3131; 
testLen = 1000; 
initLen = 100; 
data=load('data2#.txt');
x=data(:,[1:2]);
y=data(:,3);

% generate the ESN reservoir 
inSize = 2; %输入维度K
outSize = 1;
resSize = 800; 

%输入输出归一化处理
[x0,ps1]=mapminmax(x',0,1);
x0=x0';
[y0,ps2]=mapminmax(y',0,1);
y0=y0';
trainInputSequence=x0((1:trainLen),:);
trainOutputSequence=y0((1:trainLen),:);

testInputSequence=x0((trainLen+1:trainLen+testLen),:);
testOutputSequence=y0((trainLen+1:trainLen+testLen),:);

esn = generate_esn(inSize, resSize,outSize, ...
   'spectralRadius',0.2,'inputScaling',[1;1],'inputShift',[0;0], ...
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
nForgetPoints =10 ; 
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
plot(testOutputSequence(1:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen),'b.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('测试教师数据 (红色实线) vs 预测数据 (蓝色实心点)');
xlim([0,200]);
figure(12);
%nPlotPoints = 100 ; 
plot(testOutputSequence(1:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen),'b.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('测试教师数据 (红色实线) vs 预测数据 (蓝色实心点)');
xlim([401,testLen]);
%%%%compute training error

%%%%计算训练误差
[NRMSE,Absolute_err,less1,less2,more2] = compute_error(predictedTrainOutput, trainOutputSequence);
%trainError = compute_error(predictedTrainOutput, trainInputSequence); 
disp(sprintf('train NRMSE = %s', num2str(NRMSE)));
disp(sprintf('train Absolute_err = %s', num2str(Absolute_err)));
disp(sprintf('train less1 = %s', num2str(less1)));
disp(sprintf('train less2 = %s', num2str(less2)));
disp(sprintf('train more2 = %s', num2str(more2)));
%%%%计算测试误差
[NRMSE,Absolute_err,less1,less2,more2] = compute_error(predictedTestOutput,testOutputSequence);
%trainError = compute_error(predictedTrainOutput, trainInputSequence); 
disp(sprintf('test NRMSE = %s', num2str(NRMSE)));
disp(sprintf('test Absolute_err = %s', num2str(Absolute_err)));
disp(sprintf('test less1 = %s', num2str(less1)));
disp(sprintf('test less2 = %s', num2str(less2)));
disp(sprintf('test more2 = %s', num2str(more2)));
if 0
%画三维图显示
[x1,x2,f] = generate_GriewangK(-2.048,2.048,50);
figure(10);  
surf_sequence(x1,x2,f); 
title('2500组数据'); 

figure(13);
%测试数据和预测结果
trainInputSequence=x((1:trainLen),:);
testInputSequence=x((trainLen+1:trainLen+testLen),:);
scatter3(testInputSequence(:,1),testInputSequence(:,2),predictedTestOutput,'b.');
hold on;
scatter3(testInputSequence(:,1),testInputSequence(:,2),testOutputSequence(1:testLen),'ro');
zlim([0:1.5]);
hold off;
title('测试数据和预测结果');

figure(14);
%训练数据和预测结果
scatter3(trainInputSequence(:,1),trainInputSequence(:,2),predictedTrainOutput,'b.');
hold on;
scatter3(trainInputSequence(:,1),trainInputSequence(:,2),trainOutputSequence(1+nForgetPoints:trainLen),'ro');
zlim([0:1.5]);
hold off;
title('训练数据和预测结果');
figure(16);
scatter3(testInputSequence(:,1),testInputSequence(:,2),testOutputSequence(1+nForgetPoints:testLen),'ro');
title('测试数据');
figure(15);
scatter3(testInputSequence(:,1),testInputSequence(:,2),predictedTestOutput,'b.');
title('预测结果');
end