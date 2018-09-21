clear
trainLen = 3131; 
testLen = 1000; 
initLen = 101; 
data=load('data2#.txt');
x=data(:,[1:2]);
y=data(:,3);

% generate the ESN reservoir 
inSize = 2; %����ά��K
outSize = 1;
resSize = 300; 
x=[x1,x2];
%�����һ������Ϊx0
[x0,ps1]=mapminmax(x',0,1);
x0=x0';
trainInputSequence=x0((1:trainLen),:);
testInputSequence=x0((trainLen+1:trainLen+testLen),:);
%�����һ������Ϊt1
[y0,ps2]=mapminmax(y',0,1);
y0=y0';
trainOutputSequence=y0((1:trainLen),:);
testOutputSequence=y0((trainLen+1:trainLen+testLen),:);

esn = generate_esn(inSize, resSize,outSize, ...
   'spectralRadius',0.5,'inputScaling',[1;1],'inputShift',[0;0], ...
    'teacherScaling',[1],'teacherShift',[0],'feedbackScaling', 0, ...
    'type', 'plain_esn'); 

esn.internalWeights = esn.spectralRadius * esn.internalWeights_UnitSR;%�װ뾶Ϊ0.5
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
%ѵ��Ԥ���������һ��
t1=mapminmax('reverse',predictedTrainOutput',ps2);
predictedTrainOutput=t1';
trainOutputSequence=y((1:trainLen),:);

predictedTestOutput = test_esn(testInputSequence,  trainedEsn, nForgetPoints) ; 
%�����������һ��
t2=mapminmax('reverse',predictedTestOutput',ps2);
predictedTestOutput=t2';
testOutputSequence=y((trainLen+1:trainLen+testLen),:);

% create input-output plots
figure(11);
%�������ݵ�Ԥ��Ч��ͼ
plot(testOutputSequence(1:200),'r-');
hold on;
plot(predictedTestOutput(1:200),'b.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('���Խ�ʦ���� (��ɫʵ��) vs Ԥ������ (��ɫʵ�ĵ�)');
xlim([0,200]);
figure(12);
plot(testOutputSequence(1:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen),'b.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('���Խ�ʦ���� (��ɫʵ��) vs Ԥ������ (��ɫʵ�ĵ�)');
xlim([201,testLen]);


figure(13);
%ѵ�����ݵ�Ԥ��ͼ��
plot(trainOutputSequence(1:200),'r-');
hold on;
plot(predictedTrainOutput(1:200),'b.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('ѵ����ʦ���� (��ɫʵ��) vs Ԥ������ (��ɫʵ�ĵ�)');
xlim([0,200]);


%%%%����ѵ�����
[NRMSE,Absolute_err,less1,less2,more2] = compute_error(predictedTrainOutput, trainOutputSequence);
%trainError = compute_error(predictedTrainOutput, trainInputSequence); 
disp(sprintf('train NRMSE = %s', num2str(NRMSE)));
disp(sprintf('train Absolute_err = %s', num2str(Absolute_err)));
disp(sprintf('train less1 = %s', num2str(less1)));
disp(sprintf('train less2 = %s', num2str(less2)));
disp(sprintf('train more2 = %s', num2str(more2)));
%%%%����������
[NRMSE,Absolute_err,less1,less2,more2] = compute_error(predictedTestOutput,testOutputSequence);
%trainError = compute_error(predictedTrainOutput, trainInputSequence); 
disp(sprintf('test NRMSE = %s', num2str(NRMSE)));
disp(sprintf('test Absolute_err = %s', num2str(Absolute_err)));
disp(sprintf('test less1 = %s', num2str(less1)));
disp(sprintf('test less2 = %s', num2str(less2)));
disp(sprintf('test more2 = %s', num2str(more2)));

%����άͼ��ʾ
%[x1,x2,f] = generate_GriewangK(-2.048,2.048,50);
%figure(10);  
%surf_sequence(x2,x1,f); 
%title('2500������'); 
%hold on;
figure(14);
%�������ݵ�Ԥ��ͼ��
trainInputSequence=x((1:trainLen),:);
testInputSequence=x((trainLen+1:trainLen+testLen),:);
scatter3(testInputSequence(:,1),testInputSequence(:,2),predictedTestOutput,'b.');
hold on;
scatter3(testInputSequence(:,1),testInputSequence(:,2),testOutputSequence,'ro');
hold off;
title('�������ݺ�Ԥ��ͼ��');
figure(15);
%ѵ�����ݵ�Ԥ��ͼ��
scatter3(trainInputSequence(:,1),trainInputSequence(:,2),predictedTrainOutput,'b.');
hold on;
scatter3(trainInputSequence(:,1),trainInputSequence(:,2),trainOutputSequence,'ro');
hold off;
title('ѵ�����ݺ�Ԥ��ͼ��');

%figure(16);
%scatter3(testInputSequence(:,1),testInputSequence(:,2),testOutputSequence,'ro');
%figure(15);
%scatter3(testInputSequence(:,1),testInputSequence(:,2),predictedTestOutput,'b.');
