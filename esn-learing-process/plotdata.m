% create input-output plots
figure(11);
%测试数据的预测效果图
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
ylabel('Lowest cost');
hold off;
title('testing: teacher sequence (red) vs predicted sequence (blue)');
%title('测试教师数据 (红色实线) vs 预测数据 (蓝色虚线)');
xlim([1,testLen-nForgetPoints]);
figure(12);
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
ylabel('Lowest cost');
title('testing: teacher sequence (red) vs predicted sequence (blue)');
%title('测试教师数据 (红色实线) vs 预测数据 (蓝色虚线)');
xlim([1,100]);
figure(13);
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
title('testing: teacher sequence (red) vs predicted sequence (blue)');
%title('测试教师数据 (红色实线) vs 预测数据 (蓝色虚线)');
ylabel('Lowest cost');
xlim([101,200]);
figure(14);
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
ylabel('Lowest cost');
title('testing: teacher sequence (red) vs predicted sequence (blue)');
%title('测试教师数据 (红色实线) vs 预测数据 (蓝色虚线)');
xlim([201,300]);
figure(15);
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
ylabel('Lowest cost');
title('testing: teacher sequence (red) vs predicted sequence (blue)');
%title('测试教师数据 (红色实线) vs 预测数据 (蓝色虚线)');
xlim([301,400]);
figure(16);
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
ylabel('Lowest cost');
title('testing: teacher sequence (red) vs predicted sequence (blue)');
%title('测试教师数据 (红色实线) vs 预测数据 (蓝色虚线)');
xlim([401,500]);
figure(17);
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
ylabel('Lowest cost');
title('testing: teacher sequence (red) vs predicted sequence (blue)');
%title('测试教师数据 (红色实线) vs 预测数据 (蓝色虚线)');
xlim([501,600]);
figure(18);
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
ylabel('Lowest cost');
title('testing: teacher sequence (red) vs predicted sequence (blue)');
%title('测试教师数据 (红色实线) vs 预测数据 (蓝色虚线)');
xlim([601,700]);
figure(19);
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
ylabel('Lowest cost');
title('testing: teacher sequence (red) vs predicted sequence (blue)');
%title('测试教师数据 (红色实线) vs 预测数据 (蓝色虚线)');
xlim([701,800]);
figure(20);
plot(testOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTestOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
ylabel('Lowest cost');
title('testing: teacher sequence (red) vs predicted sequence (blue)');
%title('测试教师数据 (红色实线) vs 预测数据 (蓝色虚线)');
xlim([801,900]);



figure(21);
%训练数据的预测图形
plot(trainOutputSequence(1+nForgetPoints:trainLen),'r-');
hold on;
plot(predictedTrainOutput(1:trainLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
ylabel('Lowest cost');
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('training: teacher sequence (red) vs predicted sequence (blue)');
xlim([0,trainLen-nForgetPoints]);
figure(22);
%训练数据的预测图形
plot(trainOutputSequence(1+nForgetPoints:trainLen),'r-');
hold on;
plot(predictedTrainOutput(1:trainLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
ylabel('Lowest cost');
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('training: teacher sequence (red) vs predicted sequence (blue)');
xlim([0,100]);
figure(23);
%训练数据的预测图形
plot(trainOutputSequence(1+nForgetPoints:trainLen),'r-');
hold on;
plot(predictedTrainOutput(1:trainLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
ylabel('Lowest cost');
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('training: teacher sequence (red) vs predicted sequence (blue)');
xlim([101,200]);
figure(24);
%训练数据的预测图形
plot(trainOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTrainOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
ylabel('Lowest cost');
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('training: teacher sequence (red) vs predicted sequence (blue)');
xlim([201,300]);
figure(25);
%训练数据的预测图形
plot(trainOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTrainOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
ylabel('Lowest cost');
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('training: teacher sequence (red) vs predicted sequence (blue)');
xlim([301,400]);
figure(26);
%训练数据的预测图形
plot(trainOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTrainOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
ylabel('Lowest cost');
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('training: teacher sequence (red) vs predicted sequence (blue)');
xlim([401,500]);
figure(27);
%训练数据的预测图形
plot(trainOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTrainOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
ylabel('Lowest cost');
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('training: teacher sequence (red) vs predicted sequence (blue)');
xlim([501,600]);
figure(28);
%训练数据的预测图形
plot(trainOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTrainOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
ylabel('Lowest cost');
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('training: teacher sequence (red) vs predicted sequence (blue)');
xlim([601,700]);
figure(29);
%训练数据的预测图形
plot(trainOutputSequence(1+nForgetPoints:testLen),'r-');
hold on;
plot(predictedTrainOutput(1:testLen-nForgetPoints),'b-.');
legend('teacher sequence','predicted sequence');
hold off;
ylabel('Lowest cost');
%title('testing: teacher sequence (red) vs predicted sequence (blue)');
title('training: teacher sequence (red) vs predicted sequence (blue)');
xlim([701,800]);

figure(30);
scatter3(x(:,1),x(:,2),y,'b.');
axis([2770,3270,1.9,5.1,240000,290000]);
xlabel('Pressure');
ylabel('Water to carbon ratio');
zlabel('Lowest cost');

figure(31);

scatter3(x(:,2),x(:,1),y,'b.');
axis([1.9,5.1,2770,3270,240000,290000]);
ylabel('Pressure');
xlabel('Water to carbon ratio');
zlabel('Lowest cost');
%title('4100组数据');