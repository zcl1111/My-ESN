%clear
%求平均值
trainLen = 3000; 
testLen = 1100; 
initLen = 100; 
%data=load('data2#flow_cost.txt');
%a=randperm(trainLen+testLen);
%data=data(a,:);

x=data(:,[1:2]);%压力和水碳比
y=data(:,4);%最小成本

% generate the ESN reservoir 
inSize = 2; %输入维度K
outSize = 1;
resSize = 1500; 

if 0
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
end
for i=1:5
esn = generate_esn3(inSize, resSize,outSize, ...
   'spectralRadius',0.004,'inputScaling',[0.0001;0.01],'inputShift',[0;0], ...
    'teacherScaling',[0.0001],'teacherShift',[0],'feedbackScaling', 0, ...
    'type', 'leaky1_esn'); 

esn.internalWeights = esn.spectralRadius * esn.internalWeights_UnitSR;%谱半径为0.5
nForgetPoints = initLen;
[trainedEsn ,stateMatrix] = ...
    train_esn(trainInputSequence, trainOutputSequence, esn, nForgetPoints) ;

%%%% plot the internal states of 4 units
%nPoints = 200 ; 
%plot_states(stateMatrix,[1 2 3 4], nPoints, 1) ; 

% compute the output of the trained ESN on the training and testing data,
% discarding the first nForgetPoints of each
nForgetPoints =100 ; 
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

if 0
% create input-output plots
figure(11);
%测试数据的预测效果图
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b-.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('测试教师数据 (红色实线) vs 预测数据 (蓝色实心点)');
xlim([1,testLen-nForgetPoints]);
figure(12);
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b-.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('测试教师数据 (红色实线) vs 预测数据 (蓝色实心点)');
xlim([101,201]);


figure(13);
%训练数据的预测图形
plot(trainOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTrainOutput(1:testLen-nForgetPoints),'b-.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('训练教师数据 (红色实线) vs 预测数据 (蓝色实心点)');
xlim([0,testLen-nForgetPoints]);
figure(14);
%训练数据的预测图形
plot(trainOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTrainOutput(1:testLen-nForgetPoints),'b-.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('训练教师数据 (红色实线) vs 预测数据 (蓝色实心点)');
xlim([101,201]);
end 
%%%%计算训练误差
[NRMSE1,Absolute_err1,less11,less21,more21] = compute_error(predictedTrainOutput, trainOutputSequence);
a1(i)=NRMSE1;
b1(i)=Absolute_err1;
c1(i)=less11;
d1(i)=less21;
e1(i)=more21;

[NRMSE2,Absolute_err2,less12,less22,more22] = compute_error(predictedTestOutput,testOutputSequence);
a2(i)=NRMSE2;
b2(i)=Absolute_err2;
c2(i)=less12;
d2(i)=less22;
e2(i)=more22;
end
a1=sum(a1)/5;
b1=sum(b1)/5;
c1=sum(c1)/5;
d1=sum(d1)/5;
e1=sum(e1)/5;
a2=sum(a2)/5;
b2=sum(b2)/5;
c2=sum(c2)/5;
d2=sum(d2)/5;
e2=sum(e2)/5;
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

if 0
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
axis([2770,3270,1.9,5.1,240000,290000]);
title('4100组数据');
end