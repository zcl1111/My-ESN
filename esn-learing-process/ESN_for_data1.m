%clear
trainLen =2000; 
testLen = 377; 
initLen = 100; 
data1=load('data270000less.txt');
data2=load('data270000more.txt');
a=randperm(trainLen+testLen);
data1=data1(a,:);

x=data1(:,[1:2]);%ѹ����ˮ̼��
y=data1(:,4);%��С�ɱ�

% generate the ESN reservoir 
inSize = 2; %����ά��K
outSize = 1;
resSize = 300; 

%�����һ������Ϊx0
[x0,ps1]=mapminmax(x',-1,1);
x0=x0';
trainInputSequence=x0((1:trainLen),:);
testInputSequence=x0((trainLen+1:trainLen+testLen),:);

%�����һ������Ϊt1
[y0,ps2]=mapminmax(y',-1,1);
y0=y0';
trainOutputSequence=y0((1:trainLen),:);
testOutputSequence=y0((trainLen+1:trainLen+testLen),:);

esn = generate_esn3(inSize, resSize,outSize, ...
   'spectralRadius',0.005,'inputScaling',[1;1],'inputShift',[0;0], ...
    'teacherScaling',[1],'teacherShift',[0],'feedbackScaling', 0, ...
    'type', 'leaky1_esn'); 

esn.internalWeights = esn.spectralRadius * esn.internalWeights_UnitSR;%�װ뾶Ϊ0.5
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
%ѵ��Ԥ���������һ��
t1=mapminmax('reverse',predictedTrainOutput',ps2);
predictedTrainOutput=t1';
trainOutputSequence=y((1:trainLen),:);
%figure;
%plot(trainOutputSequence);
%title('ѵ������Ŀ���������');

predictedTestOutput = test_esn(testInputSequence,  trainedEsn, nForgetPoints) ; 
%�����������һ��
t2=mapminmax('reverse',predictedTestOutput',ps2);
predictedTestOutput=t2';
testOutputSequence=y((trainLen+1:trainLen+testLen),:);
%figure(10);
%plot(testOutputSequence);
%title('��������Ŀ���������');

% create input-output plots
figure(11);
%�������ݵ�Ԥ��Ч��ͼ
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('���Խ�ʦ���� (��ɫʵ��) vs Ԥ������ (��ɫʵ�ĵ�)');
xlim([1,100]);
figure(12);
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('���Խ�ʦ���� (��ɫʵ��) vs Ԥ������ (��ɫʵ�ĵ�)');
xlim([101,testLen-nForgetPoints]);


figure(13);
%ѵ�����ݵ�Ԥ��ͼ��
plot(trainOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTrainOutput(1:testLen-nForgetPoints),'b.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('ѵ����ʦ���� (��ɫʵ��) vs Ԥ������ (��ɫʵ�ĵ�)');
xlim([0,100]);
figure(14);
%ѵ�����ݵ�Ԥ��ͼ��
plot(trainOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTrainOutput(1:testLen-nForgetPoints),'b.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('ѵ����ʦ���� (��ɫʵ��) vs Ԥ������ (��ɫʵ�ĵ�)');
xlim([101,testLen-nForgetPoints]);
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
figure(15);
%�������ݺ�Ԥ����
trainInputSequence=x((1+nForgetPoints:trainLen),:);
testInputSequence=x((trainLen+1+nForgetPoints:trainLen+testLen),:);
scatter3(testInputSequence(:,1),testInputSequence(:,2),predictedTestOutput,'b.');
hold on;
scatter3(testInputSequence(:,1),testInputSequence(:,2),testOutputSequence(1+nForgetPoints:testLen),'ro');

hold off;
title('��������(��ɫȦ)��Ԥ��������ɫʵ�ĵ㣩');
%axis([-4,4,-4,4,0,1.5]);
figure(16);
%ѵ�����ݺ�Ԥ����
scatter3(trainInputSequence(:,1),trainInputSequence(:,2),predictedTrainOutput,'b.');
hold on;
scatter3(trainInputSequence(:,1),trainInputSequence(:,2),trainOutputSequence(1+nForgetPoints:trainLen),'ro');
%axis([-4,4,-4,4,0,1.5]);
hold off;
title('ѵ������(��ɫȦ)��Ԥ��������ɫʵ�ĵ㣩');
figure(17);
scatter3(x(:,1),x(:,2),y,'b.');
title('4100������');