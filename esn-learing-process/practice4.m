clear
trainLen = 3000; 
testLen = 1097; 
initLen = 100; 
data=load('2#quchu4097.txt');
a=randperm(trainLen+testLen);
data=data(a,:);

x=data(:,[1:2]);%ѹ����ˮ̼��
y=data(:,4);%��С�ɱ�

% generate the ESN reservoir 
inSize = 2; %����ά��K
outSize = 1;
resSize = 1400; 

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
   'spectralRadius',0.004,'inputScaling',[0.0001;0.01],'inputShift',[0;0], ...
    'teacherScaling',[0.0001],'teacherShift',[0],'feedbackScaling', 0, ...
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
nForgetPoints =100 ; 
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
figure(10);
%�������ݵ�Ԥ��Ч��ͼ
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
title('testing: teacher sequence (red) vs predicted sequence (blue)');
%title('���Խ�ʦ���� (��ɫʵ��) vs Ԥ������ (��ɫʵ�ĵ�)');
xlim([0,testLen-nForgetPoints]);
figure(11);
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
title('testing: teacher sequence (red) vs predicted sequence (blue)');
%title('���Խ�ʦ���� (��ɫʵ��) vs Ԥ������ (��ɫʵ�ĵ�)');
xlim([0,100]);
figure(12);
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
title('testing: teacher sequence (red) vs predicted sequence (blue)');
%title('���Խ�ʦ���� (��ɫʵ��) vs Ԥ������ (��ɫʵ�ĵ�)');
xlim([100,201]);

figure(13);
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
title('testing: teacher sequence (red) vs predicted sequence (blue)');
%title('���Խ�ʦ���� (��ɫʵ��) vs Ԥ������ (��ɫʵ�ĵ�)');
xlim([200,301]);
figure(14);
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
title('testing: teacher sequence (red) vs predicted sequence (blue)');
%title('���Խ�ʦ���� (��ɫʵ��) vs Ԥ������ (��ɫʵ�ĵ�)');
xlim([300,401]);
figure(15);
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
title('testing: teacher sequence (red) vs predicted sequence (blue)');
%title('���Խ�ʦ���� (��ɫʵ��) vs Ԥ������ (��ɫʵ�ĵ�)');
xlim([400,601]);
figure(16);
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
title('testing: teacher sequence (red) vs predicted sequence (blue)');
%title('���Խ�ʦ���� (��ɫʵ��) vs Ԥ������ (��ɫʵ�ĵ�)');
xlim([600,701]);
figure(17);
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
title('testing: teacher sequence (red) vs predicted sequence (blue)');
%title('���Խ�ʦ���� (��ɫʵ��) vs Ԥ������ (��ɫʵ�ĵ�)');
xlim([700,801]);
figure(18);
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
title('testing: teacher sequence (red) vs predicted sequence (blue)');
%title('���Խ�ʦ���� (��ɫʵ��) vs Ԥ������ (��ɫʵ�ĵ�)');
xlim([800,901]);

figure(21);
%ѵ�����ݵ�Ԥ��ͼ��
plot(trainOutputSequence(1+nForgetPoints:trainLen),'r-');
hold on;
plot(predictedTrainOutput(1:trainLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
title('training: teacher sequence (red) vs predicted sequence (blue)');
%title('ѵ����ʦ���� (��ɫʵ��) vs Ԥ������ (��ɫʵ�ĵ�)');
xlim([0,trainLen-nForgetPoints]);
figure(22);
%ѵ�����ݵ�Ԥ��ͼ��
plot(trainOutputSequence(1+nForgetPoints:trainLen),'r-');
hold on;
plot(predictedTrainOutput(1:trainLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
title('training: teacher sequence (red) vs predicted sequence (blue)');
xlim([0,100]);
figure(23);
%ѵ�����ݵ�Ԥ��ͼ��
plot(trainOutputSequence(1+nForgetPoints:trainLen),'r-');
hold on;
plot(predictedTrainOutput(1:trainLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
title('training: teacher sequence (red) vs predicted sequence (blue)');
xlim([100,200]);
figure(23);
%ѵ�����ݵ�Ԥ��ͼ��
plot(trainOutputSequence(1+nForgetPoints:trainLen),'r-');
hold on;
plot(predictedTrainOutput(1:trainLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
title('training: teacher sequence (red) vs predicted sequence (blue)');
xlim([200,300]);
figure(24);
%ѵ�����ݵ�Ԥ��ͼ��
plot(trainOutputSequence(1+nForgetPoints:trainLen),'r-');
hold on;
plot(predictedTrainOutput(1:trainLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
title('training: teacher sequence (red) vs predicted sequence (blue)');
xlim([300,400]);
figure(25);
%ѵ�����ݵ�Ԥ��ͼ��
plot(trainOutputSequence(1+nForgetPoints:trainLen),'r-');
hold on;
plot(predictedTrainOutput(1:trainLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
title('training: teacher sequence (red) vs predicted sequence (blue)');
xlim([400,500]);
figure(26);
%ѵ�����ݵ�Ԥ��ͼ��
plot(trainOutputSequence(1+nForgetPoints:trainLen),'r-');
hold on;
plot(predictedTrainOutput(1:trainLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
title('training: teacher sequence (red) vs predicted sequence (blue)');
xlim([500,600]);
figure(27);
%ѵ�����ݵ�Ԥ��ͼ��
plot(trainOutputSequence(1+nForgetPoints:trainLen),'r-');
hold on;
plot(predictedTrainOutput(1:trainLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
title('training: teacher sequence (red) vs predicted sequence (blue)');
xlim([600,700]);
%%%%����ѵ�����
[NRMSE,Absolute_err,less1,less2,more2] = compute_error3(predictedTrainOutput, trainOutputSequence);
%trainError = compute_error(predictedTrainOutput, trainInputSequence); 
disp(sprintf('train NRMSE = %s', num2str(NRMSE)));
disp(sprintf('train Absolute_err = %s', num2str(Absolute_err)));
disp(sprintf('train less1 = %s', num2str(less1)));
disp(sprintf('train less2 = %s', num2str(less2)));
disp(sprintf('train more2 = %s', num2str(more2)));
%%%%����������
[NRMSE,Absolute_err,less1,less2,more2] = compute_error3(predictedTestOutput,testOutputSequence);
%trainError = compute_error(predictedTrainOutput, trainInputSequence); 
disp(sprintf('test NRMSE = %s', num2str(NRMSE)));
disp(sprintf('test Absolute_err = %s', num2str(Absolute_err)));
disp(sprintf('test less1 = %s', num2str(less1)));
disp(sprintf('test less2 = %s', num2str(less2)));
disp(sprintf('test more2 = %s', num2str(more2)));
if 0
figure(15);
%�������ݺ�Ԥ����
trainInputSequence=x((1+nForgetPoints:trainLen),:);
testInputSequence=x((trainLen+1+nForgetPoints:trainLen+testLen),:);
scatter3(testInputSequence(:,1),testInputSequence(:,2),predictedTestOutput,'b.');
hold on;
scatter3(testInputSequence(:,1),testInputSequence(:,2),testOutputSequence(1+nForgetPoints:testLen),'ro');
legend('teacher sequence','predicted sequence');
hold off;
%title('��������(��ɫȦ)��Ԥ��������ɫʵ�ĵ㣩');
%axis([-4,4,-4,4,0,1.5]);
figure(16);
%ѵ�����ݺ�Ԥ����
scatter3(trainInputSequence(:,1),trainInputSequence(:,2),predictedTrainOutput,'b.');
hold on;
scatter3(trainInputSequence(:,1),trainInputSequence(:,2),trainOutputSequence(1+nForgetPoints:trainLen),'ro');
legend('teacher sequence','predicted sequence');
%axis([-4,4,-4,4,0,1.5]);
hold off;
%title('ѵ������(��ɫȦ)��Ԥ��������ɫʵ�ĵ㣩');
end
figure(17);
scatter3(x(:,1),x(:,2),y,'b.');
axis([2770,3270,1.9,5.1,240000,290000]);
%title('4100������');