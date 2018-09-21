%clear
trainLen =2000; 
testLen = 377; 
initLen = 100; 
data1=load('data270000less.txt');
data2=load('data270000more.txt');
a=randperm(trainLen+testLen);
data1=data1(a,:);

x=data1(:,[1:2]);%压力和水碳比
y=data1(:,4);%最小成本

% generate the ESN reservoir 
inSize = 2; %输入维度K
outSize = 1;
resSize = 300; 

%输入归一化处理为x0
[x0,ps1]=mapminmax(x',-1,1);
x0=x0';
trainInputSequence=x0((1:trainLen),:);
testInputSequence=x0((trainLen+1:trainLen+testLen),:);

%输出归一化处理为t1
[y0,ps2]=mapminmax(y',-1,1);
y0=y0';
trainOutputSequence=y0((1:trainLen),:);
testOutputSequence=y0((trainLen+1:trainLen+testLen),:);

esn = generate_esn3(inSize, resSize,outSize, ...
   'spectralRadius',0.005,'inputScaling',[1;1],'inputShift',[0;0], ...
    'teacherScaling',[1],'teacherShift',[0],'feedbackScaling', 0, ...
    'type', 'leaky1_esn'); 

esn.internalWeights = esn.spectralRadius * esn.internalWeights_UnitSR;%谱半径为0.5
nForgetPoints = initLen;
[trainedEsn ,stateMatrix] = ...
    train_esn(trainInputSequence, trainOutputSequence, esn, nForgetPoints) ;

%%%% plot the internal states of 4 units
nPoints = 200 ; 
plot_states(stateMatrix,[1 2 3 4], nPoints, 1) ; 

% compute the output of the trained ESN on the training and testing data,
% discarding the first nForgetPoints of each
nForgetPoints =50; 
predictedTrainOutput = test_esn(trainInputSequence, trainedEsn, nForgetPoints);
%训练预测输出反归一化
t1=mapminmax('reverse',predictedTrainOutput',ps2);
predictedTrainOutput=t1';
trainOutputSequence=y((1:trainLen),:);
%figure;
%plot(trainOutputSequence);
%title('训练数据目标输出序列');

predictedTestOutput = test_esn(testInputSequence,  trainedEsn, nForgetPoints) ; 
%测试输出反归一化
t2=mapminmax('reverse',predictedTestOutput',ps2);
predictedTestOutput=t2';
testOutputSequence=y((trainLen+1:trainLen+testLen),:);
%figure(10);
%plot(testOutputSequence);
%title('测试数据目标输出序列');

% create input-output plots
figure(11);
%测试数据的预测效果图
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('测试教师数据 (红色实线) vs 预测数据 (蓝色实心点)');
xlim([1,100]);
figure(12);
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('测试教师数据 (红色实线) vs 预测数据 (蓝色实心点)');
xlim([101,testLen-nForgetPoints]);


figure(13);
%训练数据的预测图形
plot(trainOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTrainOutput(1:testLen-nForgetPoints),'b.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('训练教师数据 (红色实线) vs 预测数据 (蓝色实心点)');
xlim([0,100]);
figure(14);
%训练数据的预测图形
plot(trainOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTrainOutput(1:testLen-nForgetPoints),'b.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('训练教师数据 (红色实线) vs 预测数据 (蓝色实心点)');
xlim([101,testLen-nForgetPoints]);
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
figure(15);
%测试数据和预测结果
trainInputSequence=x((1+nForgetPoints:trainLen),:);
testInputSequence=x((trainLen+1+nForgetPoints:trainLen+testLen),:);
scatter3(testInputSequence(:,1),testInputSequence(:,2),predictedTestOutput,'b.');
hold on;
scatter3(testInputSequence(:,1),testInputSequence(:,2),testOutputSequence(1+nForgetPoints:testLen),'ro');

hold off;
title('测试数据(红色圈)和预测结果（蓝色实心点）');
%axis([-4,4,-4,4,0,1.5]);
figure(16);
%训练数据和预测结果
scatter3(trainInputSequence(:,1),trainInputSequence(:,2),predictedTrainOutput,'b.');
hold on;
scatter3(trainInputSequence(:,1),trainInputSequence(:,2),trainOutputSequence(1+nForgetPoints:trainLen),'ro');
%axis([-4,4,-4,4,0,1.5]);
hold off;
title('训练数据(红色圈)和预测结果（蓝色实心点）');
figure(17);
scatter3(x(:,1),x(:,2),y,'b.');
title('4100组数据');