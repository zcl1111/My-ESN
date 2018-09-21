data=load('data2#flow_cost.txt');
a=find(data(:,4)<270000);
b=find(data(:,4)>270000);
data1=data(a,:);
data2=data(b,:);
