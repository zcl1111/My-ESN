trainLen = 3000; 
testLen = 1097; 
initLen = 100; 
data=load('2#quchu4097.txt');

a=randperm(trainLen+testLen);
data=data(a,:);
x=data(:,[1:2]);
y=data(:,4);
[x0,ps1]=mapminmax(x',0,1);
%x0=x0';
[y0,ps2]=mapminmax(y',0,1);
%y0=y0';
trainInputSequence=x0(:,(1:trainLen));
trainOutputSequence=y0(:,(1:trainLen));

testInputSequence=x0(:,(trainLen+1:trainLen+testLen));
testOutputSequence=y0(:,(trainLen+1:trainLen+testLen));

net=newff(minmax(trainInputSequence),[2,30,1],{'tansig','tansig','purelin'},'trainlm');
% Ñ§Ï°ËÙÂÊÎª0.05¡£
net.trainParam.lr=0.05;
%ÉèÖÃÑµÁ·´ÎÊı
net.trainParam.epochs =1000;
%ÉèÖÃÊÕÁ²Îó²î
%net.trainParam.goal=0.0000001;
%ÑµÁ·ÍøÂç
[net,tr]=train(net,trainInputSequence,trainOutputSequence);
%ÑµÁ·Êä³öÖµ
y1=sim(net,trainInputSequence);
t1=mapminmax('reverse',y1,ps2);
predictedTrainOutput=t1';
trainOutputSequence=y((1:trainLen),:);

%²âÊÔÊä³öÖµ
y2=sim(net,testInputSequence);
t2=mapminmax('reverse',y2,ps2);
predictedTestOutput=t2';
testOutputSequence=y((trainLen+1:trainLen+testLen),:);
figure(21);
plot(testOutputSequence,'r-');
hold on;
plot(predictedTestOutput,'b.');
hold off;  
xlim([1,200]);
figure(22);
plot(testOutputSequence,'r-');
hold on;
plot(predictedTestOutput,'b.');
hold off; 
xlim([201,400]);
figure(23);
plot(testOutputSequence,'r-');
hold on;
plot(predictedTestOutput,'b.');
hold off; 
xlim([401,1000]);
%%%%¼ÆËãÑµÁ·Îó²î
[NRMSE,Absolute_err] = compute_error1(predictedTrainOutput, trainOutputSequence);
[mean_Relative_err,less1,less2,more2] = compute_error2(predictedTrainOutput, trainOutputSequence);
%trainError = compute_error(predictedTrainOutput, trainInputSequence); 
disp(sprintf('train NRMSE = %s', num2str(NRMSE)));
disp(sprintf('train Absolute_err = %s', num2str(Absolute_err)));
disp(sprintf('train mean_Relative_err=%s',num2str(mean_Relative_err)));
disp(sprintf('train less1 = %s', num2str(less1)));
disp(sprintf('train less2 = %s', num2str(less2)));
disp(sprintf('train more2 = %s', num2str(more2)));
%%%%¼ÆËã²âÊÔÎó²î
[NRMSE,Absolute_err] = compute_error1(predictedTestOutput,testOutputSequence);
[mean_Relative_err,less1,less2,more2] = compute_error2(predictedTestOutput,testOutputSequence);
%trainError = compute_error(predictedTrainOutput, trainInputSequence); 
disp(sprintf('test NRMSE = %s', num2str(NRMSE)));
disp(sprintf('test Absolute_err = %s', num2str(Absolute_err)));
disp(sprintf('train mean_Relative_err=%s',num2str(mean_Relative_err)));
disp(sprintf('test less1 = %s', num2str(less1)));
disp(sprintf('test less2 = %s', num2str(less2)));
disp(sprintf('test more2 = %s', num2str(more2)));

