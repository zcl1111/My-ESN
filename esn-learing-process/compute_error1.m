function [NRMSE,Absolute_err] = compute_error1(estimatedOutput, correctOutput)
%���������С��0.01������0.01С��0.02������0.02�ı���
nEstimatePoints = length(estimatedOutput) ; 

nForgetPoints = length(correctOutput) - nEstimatePoints ; 

correctOutput = correctOutput(nForgetPoints+1:end,:) ; 
A=abs(estimatedOutput - correctOutput);%�������
correctVariance = var(correctOutput) ; %��������ķ���
meanerror = sum((A).^2)/nEstimatePoints ; 
MSE=sqrt(meanerror);
NRMSE= (sqrt(meanerror./correctVariance)) ; %��һ�����������

Absolute_err=sum(A)/nEstimatePoints;%ƽ���������

%Relative_err=A./correctOutput;%������
%mean_Relative_err=Absolute_err/nEstimatePoints;%ƽ��������

%less1=(length(find(Relative_err<=0.01)))/nEstimatePoints;
%less2=(length(find(Relative_err<=0.02))-length(find(Relative_err<=0.01)))/nEstimatePoints;
%more2=1-less1-less2;


end

