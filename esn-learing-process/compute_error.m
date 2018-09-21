function [NRMSE,Absolute_err,less1,less2,more2] = compute_error(estimatedOutput, correctOutput)
% Computes the NRMSE between estimated and correct ESN outputs.
% 
% input arguments:
% estimatedOutput: array of size N1 x outputDimension, containing network
% output data. Caution: it is assumed that these are un-rescaled and
% un-shifted, that is, the transformations from the original data format
% via esn.teacherScaling and esn.teacherShift are undone. This happens
% automatically when the estimatedOutput was obtained from calling
% test_esn.
%
% correctOutput: array of size N2 x outputDimension, containing the
% original teacher data. 
%
% output:
% err: a row vector of NRMSE's, each corresponding to one of the output
% dimensions.
%
% If length(correctOutput) > length(estimatedOutput), the first
% elements from correctOutput are deleted. This accounts for cases where
% some (nForgetPoints many) initial transient data points were cancelled
% from estimatedOutput, as occurs in calls to test_esn.
%
% Version 1.0, June 6, 2006, H. Jaeger
% Copyright: Fraunhofer IAIS 2006 / Patents pending
  
  
nEstimatePoints = length(estimatedOutput) ; 

nForgetPoints = length(correctOutput) - nEstimatePoints ; 

correctOutput = correctOutput(nForgetPoints+1:end,:) ;

A=abs(estimatedOutput - correctOutput);%绝对误差
correctVariance = var(correctOutput) ; %期望输出的方差
meanerror = sum((A).^2)/nEstimatePoints ; 
%MSE=sqrt(meanerror);
NRMSE= (sqrt(meanerror./correctVariance)) ; %归一化均方根误差

Absolute_err=sum(A)/nEstimatePoints;%平均绝对误差

Relative_err=A./correctOutput;%相对误差

less1=(length(find(Relative_err<=0.01)))/nEstimatePoints;
less2=(length(find(Relative_err<=0.02))-length(find(Relative_err<=0.01)))/nEstimatePoints;
more2=1-less1-less2;

