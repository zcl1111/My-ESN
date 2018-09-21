clear

%clear all; 
trainLen = 2000; 
testLen = 500; 
initLen = 100; 
[x1,x2,f] = generate_GriewangK(-2.048,2.048,50);
% plot some of it 
figure(10); 
surf_sequence(x2,x1,f); 
title('2500组数据'); 
% generate the ESN reservoir 
inSize = 2; %输入维度K
outSize = 1;
resSize = 300; 
[x,y]=convert(x1,x2,f);

trainInputSequence=x((1:2000),:);
%训练输入归一化处理
x1=trainInputSequence(:,1)';
x2=trainInputSequence(:,2)';
[p1,PS1]=mapminmax(x1);
[p2,PS2]=mapminmax(x2);
trainInputSequence=[p1',p2'];

trainOutputSequence=y((1:2000),:);
%训练输出归一化处理
[t1,PS3]=mapminmax(trainOutputSequence');
trainOutputSequence=t1';

testInputSequence=x((2001:2500),:);
%测试输入归一化
x11=testInputSequence(:,1)';
x22=testInputSequence(:,2)';
[p11]=mapminmax('apply',x11,PS1);
[p22]=mapminmax('apply',x22,PS1);
trainInputSequence=[p11',p22'];

testOutputSequence=y((2001:2500),:);

esn = generate_esn(inSize, resSize,outSize, ...
   'spectralRadius',0.5,'inputScaling',[1;1],'inputShift',[0;0], ...
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
nForgetPoints =0 ; 
predictedTrainOutput = test_esn(trainInputSequence, trainedEsn, nForgetPoints);
%训练预测输出反归一化
t11=mapminmax('reverse',predictedTrainOutput',PS3);
predictedTrainOutput=t11';
trainInputSequence=x((1:2000),:);

predictedTestOutput = test_esn(testInputSequence,  trainedEsn, nForgetPoints) ; 
%测试输出反归一化
t22=mapminmax('reverse',predictedTestOutput',PS2);
predictedTestOutput=t22';

% create input-output plots
figure;
%nPlotPoints = 100 ; 
plot(testOutputSequence(101:300),'r-');
hold on;
plot(predictedTestOutput(101:300),'b.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('测试教师数据 (红色实线) vs 预测数据 (蓝色实心点)');

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
