%用来调试
clc
%clear all; 
trainLen = 300; 
testLen = 50; 
initLen = 20; 
[x1,x2,f] = generate_GriewangK(-600,600,350);
% plot some of it 
figure(10); 
surf_sequence(x1,x2,f); 
title('350组数据'); 
% generate the ESN reservoir 
inSize = 2; %输入维度K
outSize = 1;
resSize = 100; 
[x,y]=convert(x1,x2,f);
A=randperm(350);


trainInputSequence=x(A(1:300),:);
trainOutputSequence=y(A(1:300),:);
testInputSequence=x(A(301:350),:);
testOutoutSequence=y(A(301:350),:);

esn = generate_esn(inSize, resSize,outSize, ...
   'spectralRadius',0.5,'inputScaling',[0.1;0.1],'inputShift',[0;0], ...
    'teacherScaling',[0.3],'teacherShift',[-0.2],'feedbackScaling', 0, ...
    'type', 'plain_esn'); 

esn.internalWeights = esn.spectralRadius * esn.internalWeights_UnitSR;%谱半径为0.5
nForgetPoints = 20;
[trainedEsn ,stateMatrix] = ...
    train_esn(trainInputSequence, trainOutputSequence, esn, nForgetPoints) ;

%%%% plot the internal states of 4 units
nPoints = 200 ; 
plot_states(stateMatrix,[1 2 3 4], nPoints, 1) ; 

% compute the output of the trained ESN on the training and testing data,
% discarding the first nForgetPoints of each
nForgetPoints = 100 ; 
predictedTrainOutput = test_esn(trainInputSequence, trainedEsn, nForgetPoints);
predictedTestOutput = test_esn(testInputSequence,  trainedEsn, 0) ; 