
%clear all; 
trainLen =600; 
testLen = 400; 
initLen =50; 
data=normrnd(2000,10,[1000,1]);

% plot some of it 
figure(10); 
plot(data(1:600)); 
%%title('A sample of data'); 
% generate the ESN reservoir 
inSize = 1; %输入维度K
outSize = 1;
resSize = 200; 

trainInputSequence=data(1:600);
trainOutputSequence=data(1:600);
testInputSequence=data(601:1000);
testOutputSequence=data(601:1000);

%运行50次求平均值
for i=1:50
esn = generate_esn(inSize, resSize,outSize, ...
    'spectralRadius',0.8,'inputScaling',0.1,'inputShift',0, ...
    'teacherScaling',0.3,'teacherShift',0,'feedbackScaling', 0, ...
    'type', 'plain_esn'); 

esn.internalWeights =esn.spectralRadius * esn.internalWeights_UnitSR;%0.5为谱半径
nForgetPoints = 50 ;
[trainedEsn ,stateMatrix] = ...
    train_esn(trainInputSequence, trainOutputSequence, esn, nForgetPoints) ;

%%%% plot the internal states of 4 units

%nPoints =200 ; 
%plot_states(stateMatrix,[1 2 3 4], nPoints, 1) ; 

% compute the output of the trained ESN on the training and testing data,
% discarding the first nForgetPoints of each
nForgetPoints = 50 ; 
predictedTrainOutput = test_esn(trainInputSequence, trainedEsn, nForgetPoints);
predictedTestOutput = test_esn(testInputSequence,  trainedEsn,0) ; 

%%%%计算训练误差
[NRMSE1,Absolute_err1,less11,less21,more21] = compute_error(predictedTrainOutput, trainOutputSequence);
%trainError = compute_error(predictedTrainOutput, trainInputSequence); 
a1(i)=NRMSE1;
b1(i)=Absolute_err1;
c1(i)=less11;
d1(i)=less21;
e1(i)=more21;

%计算测试误差
[NRMSE2,Absolute_err2,less12,less22,more22] = compute_error(predictedTestOutput,testOutputSequence);
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
%%%%compute testing error
%[NRMSE,Absolute_err,less12,less22,more21] = compute_error(predictedTestOutput,testOutputSequence);
%trainError = compute_error(predictedTrainOutput, trainInputSequence); 
disp(sprintf('test NRMSE = %s', num2str(a2)));
disp(sprintf('test Absolute_err = %s', num2str(b2)));
disp(sprintf('test less1 = %s', num2str(c2)));
disp(sprintf('test less2 = %s', num2str(d2)));
disp(sprintf('test more2 = %s', num2str(e2)));

%testError = compute_error(predictedTestOutput, testInputSequence); 
%disp(sprintf('test error = %s', num2str(testError)))

%disp(sprintf('test error = %s', num2str(testError)))




%单独运行一次的程序
if 0
esn = generate_esn(inSize, resSize,outSize, ...
    'spectralRadius',0.8,'inputScaling',0.1,'inputShift',0, ...
    'teacherScaling',0.3,'teacherShift',0,'feedbackScaling', 0, ...
    'type', 'plain_esn'); 

esn.internalWeights =esn.spectralRadius * esn.internalWeights_UnitSR;%0.5为谱半径
nForgetPoints = 50 ;
[trainedEsn ,stateMatrix] = ...
    train_esn(trainInputSequence, trainOutputSequence, esn, nForgetPoints) ;

% plot the internal states of 4 units

nPoints =200 ; 
plot_states(stateMatrix,[1 2 3 4], nPoints, 1) ; 

% compute the output of the trained ESN on the training and testing data,
% discarding the first nForgetPoints of each
nForgetPoints = 50 ; 
predictedTrainOutput = test_esn(trainInputSequence, trainedEsn, nForgetPoints);
predictedTestOutput = test_esn(testInputSequence,  trainedEsn,0) ; 

% create input-output plots
figure(11);
%nPlotPoints = 100 ; 
plot(testInputSequence(1:200),'r-');
hold on;
plot(predictedTestOutput(1:200),'b.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('测试教师数据 (红色实线) vs 预测数据 (蓝色实心点)');
figure(12);
plot(testInputSequence(1:400),'r-');
hold on;
plot(predictedTestOutput(1:400),'b.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('测试教师数据 (红色实线) vs 预测数据 (蓝色实心点)');
xlim([200,400]);
%计算训练误差
[NRMSE,Absolute_err,less1,less2,more2] = compute_error(predictedTrainOutput, trainOutputSequence);
disp(sprintf('train NRMSE = %s', num2str(NRMSE)));
disp(sprintf('train Absolute_err = %s', num2str(Absolute_err)));
disp(sprintf('train less1 = %s', num2str(less1)));
disp(sprintf('train less2 = %s', num2str(less2)));
disp(sprintf('train more2 = %s', num2str(more2)));
%计算测试误差
[NRMSE,Absolute_err,less1,less2,more2] = compute_error(predictedTestOutput,testOutputSequence); 
disp(sprintf('test NRMSE = %s', num2str(NRMSE)));
disp(sprintf('test Absolute_err = %s', num2str(Absolute_err)));
disp(sprintf('test less1 = %s', num2str(less1)));
disp(sprintf('test less2 = %s', num2str(less2)));
disp(sprintf('test more2 = %s', num2str(more2)));
end