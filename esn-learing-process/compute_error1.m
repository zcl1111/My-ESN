function [NRMSE,Absolute_err] = compute_error1(estimatedOutput, correctOutput)
%求相对误差和小于0.01，大于0.01小于0.02，大于0.02的比例
nEstimatePoints = length(estimatedOutput) ; 

nForgetPoints = length(correctOutput) - nEstimatePoints ; 

correctOutput = correctOutput(nForgetPoints+1:end,:) ; 
A=abs(estimatedOutput - correctOutput);%绝对误差
correctVariance = var(correctOutput) ; %期望输出的方差
meanerror = sum((A).^2)/nEstimatePoints ; 
MSE=sqrt(meanerror);
NRMSE= (sqrt(meanerror./correctVariance)) ; %归一化均方根误差

Absolute_err=sum(A)/nEstimatePoints;%平均绝对误差

%Relative_err=A./correctOutput;%相对误差
%mean_Relative_err=Absolute_err/nEstimatePoints;%平均相对误差

%less1=(length(find(Relative_err<=0.01)))/nEstimatePoints;
%less2=(length(find(Relative_err<=0.02))-length(find(Relative_err<=0.01)))/nEstimatePoints;
%more2=1-less1-less2;


end

