clc
%clear all; 
trainLen = 2000; 
testLen = 2000; 
initLen = 100; 
data = load('MackeyGlass_t17.txt');
% plot some of it 
figure(10); 
plot(data(1:2000)); 
title('ǰ2000��ѵ������'); 
% generate the ESN reservoir 
inSize = 1; %����ά��K
outSize = 1;
resSize = 700; 

trainInputSequence=data(1:2000);
trainOutputSequence=data(1:2000);
testInputSequence=data(2001:4000);
testOutputSequence=data(2001:4000);
if 0
%����50����ƽ��ֵ
for i=1:50
esn = generate_esn(inSize, resSize,outSize, ...
    'spectralRadius',0.7,'inputScaling',0.1,'inputShift',0, ...
    'teacherScaling',0.3,'teacherShift',0,'feedbackScaling', 0, ...
    'type', 'plain_esn'); 

esn.internalWeights = esn.spectralRadius * esn.internalWeights_UnitSR;%�װ뾶
nForgetPoints = 100;
[trainedEsn ,stateMatrix] = ...
    train_esn(trainInputSequence, trainOutputSequence, esn, nForgetPoints) ;

%%%% plot the internal states of 4 units
%nPoints = 200 ; 
%plot_states(stateMatrix,[1 2 3 4], nPoints, 1) ;


% compute the output of the trained ESN on the training and testing data,
% discarding the first nForgetPoints of each
nForgetPoints = 100 ; 
predictedTrainOutput = test_esn(trainInputSequence, trainedEsn, nForgetPoints);
predictedTestOutput = test_esn(testInputSequence,  trainedEsn, 0) ; 

% create input-output plots
%plot_sequence(trainOutputSequence(nForgetPoints+1:end,:), predictedTrainOutput, nPlotPoints,...
   % 'ѵ����ʦ���� (����) vs Ԥ������ (�̵�)');
%plot_sequence(testOutputSequence(nForgetPoints+1:end,:), predictedTestOutput, nPlotPoints, ...
   % '���Խ�ʦ���� (����) vs Ԥ������ (�̵�)') ; 
%figure(11);
%nPlotPoints = 100 ; 
%plot(testInputSequence(1:1000),'r-');
%hold on;
%plot(predictedTestOutput(1:1000),'b.');
%hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
%title('���Խ�ʦ���� (��ɫʵ��) vs Ԥ������ (��ɫʵ�ĵ�)');
%figure(12);
%plot(testInputSequence(1:2000),'r-');
%hold on;
%plot(predictedTestOutput(1:2000),'b.');
%hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
%title('���Խ�ʦ���� (��ɫʵ��) vs Ԥ������ (��ɫʵ�ĵ�)');
%xlim([1000,2000]);

%%%%����ѵ�����
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

disp(sprintf('test NRMSE = %s', num2str(a2)));
disp(sprintf('test Absolute_err = %s', num2str(b2)));
disp(sprintf('test less1 = %s', num2str(c2)));
disp(sprintf('test less2 = %s', num2str(d2)));
disp(sprintf('test more2 = %s', num2str(e2)));
end


%��������һ�λ�ͼ
%if 0
esn = generate_esn(inSize, resSize,outSize, ...
    'spectralRadius',0.7,'inputScaling',0.1,'inputShift',0, ...
    'teacherScaling',0.3,'teacherShift',0,'feedbackScaling', 0, ...
    'type', 'plain_esn'); 

esn.internalWeights = esn.spectralRadius * esn.internalWeights_UnitSR;%�װ뾶
nForgetPoints = 100;
[trainedEsn ,stateMatrix] = ...
    train_esn(trainInputSequence, trainOutputSequence, esn, nForgetPoints) ;

% plot the internal states of 4 units
nPoints = 200 ; 
plot_states(stateMatrix,[1 2 3 4], nPoints, 1) ; 

% compute the output of the trained ESN on the training and testing data,
% discarding the first nForgetPoints of each
nForgetPoints = 0; 
predictedTrainOutput = test_esn(trainInputSequence, trainedEsn, nForgetPoints);
predictedTestOutput = test_esn(testInputSequence,  trainedEsn,nForgetPoints) ; 

% create input-output plots
figure(11);
%nPlotPoints = 100 ; 
plot(testInputSequence(1:1000),'r-');
hold on;
plot(predictedTestOutput(1:1000),'b.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('���Խ�ʦ���� (��ɫʵ��) vs Ԥ������ (��ɫʵ�ĵ�)');
figure(12);
plot(testInputSequence(1:2000),'r-');
hold on;
plot(predictedTestOutput(1:2000),'b.');
hold off;
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('���Խ�ʦ���� (��ɫʵ��) vs Ԥ������ (��ɫʵ�ĵ�)');
xlim([1000,2000]);

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
%end 