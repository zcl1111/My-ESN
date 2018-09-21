clc
clear
%分析处理氢气网络数据
data=load('2#quchu4097.txt');
%a=randperm(trainLen+testLen);
%data1=data(a,:);
figure(1);
scatter3(data(:,1),data(:,2),data(:,4),'b.');
xlabel('Pressure');
ylabel('Water-carbon ratio');
zlabel('Lowest cost');

data1=data(data(:,1)<2845,:);%根据压力节点，压力小于2845的数据
%%第一部分
data11=data1(data1(:,4)<250000,:);%为了区分块的颜色，选出成本小于250000的区域%%且是水碳比3.946165-5.079222
a=find(data11(:,4)==min(data11(:,4)));
point1=data11(a,:);

%%第二部分
data12=data1(data1(:,4)>250000&data1(:,4)<260000,:);%选出成本250000<*<260000的区域%且是水碳比3.203817-3.868023
a=find(data12(:,4)==min(data12(:,4)));
point2=data12(a,:);
%%第三部分
data31=data1(data1(:,2)>2.50&data1(:,2)<3.126,:);%选出水碳比2.50054-3.125675
data32=data1(data1(:,1)<=2840&data1(:,1)>=2800&data1(:,2)<3.17&data1(:,2)>3.16,:);%选出水碳比为3.164746，压力为2800-2840
data33=data1(data1(:,1)<=2780&data1(:,1)>=2770&data1(:,2)<2.47&data1(:,2)>2.46,:);%选出水碳比为2.461469，压力为2770-2780
data13=[data31;data32;data33];
a=find(data13(:,4)==min(data13(:,4)));
point3=data13(a,:);
%%第四部分
data41=data1(data1(:,2)>2.26&data1(:,2)<2.43,:);%选出水碳比2.266115-2.422398
data42=data1(data1(:,1)<=2840&data1(:,1)>=2790&data1(:,2)<2.47&data1(:,2)>2.46,:);%选出水碳比为2.461469，压力为2790-2840
data43=data1(data1(:,1)<=2800&data1(:,1)>=2770&data1(:,2)<2.23&data1(:,2)>2.22,:);%选出水碳比为2.227044，压力为2770-2800
data14=[data41;data42;data43];
a=find(data14(:,4)==min(data14(:,4)));
point4=data14(a,:);
%%第五部分
data51=data1(data1(:,2)>2.03&data1(:,2)<2.188,:);%选出水碳比2.031689-2.187973
data52=data1(data1(:,1)<=2840&data1(:,1)>=2810&data1(:,2)<2.23&data1(:,2)>2.22,:);%选出水碳比为2.227044，压力为2810-2840
data15=[data51;data52];
a=find(data15(:,4)==min(data15(:,4)));
point5=data15(a,:);
%%第六部分
data16=data1(data1(:,2)>1.95&data1(:,2)<2,:);%选出水碳比1.953547-1.992618
a=find(data16(:,4)==min(data16(:,4)));
point6=data16(a,:);

point11=data1(data1(:,1)==2790&data1(:,2)<3.13&data1(:,2)>3.12,:);
point12=data1(data1(:,1)==2790&data1(:,2)<3.17&data1(:,2)>3.16,:);

figure(2);
scatter3(data1(:,1),data1(:,2),data1(:,4),'b.');
xlabel('压力');
ylabel('水碳比');
zlabel('最低成本');

figure(3);
scatter3(data11(:,1),data11(:,2),data11(:,4),'b.');
hold on;
scatter3(data12(:,1),data12(:,2),data12(:,4),'g.');
scatter3(data13(:,1),data13(:,2),data13(:,4),'r.');
scatter3(data14(:,1),data14(:,2),data14(:,4),'c.');
scatter3(data15(:,1),data15(:,2),data15(:,4),'m.');
scatter3(data16(:,1),data16(:,2),data16(:,4),'k.');
scatter3(point11(1),point11(2),point11(4),100,'ko');%相邻跳跃部分的两个特殊点
scatter3(point12(1),point12(2),point12(4),100,'ko');

scatter3(point1(1),point1(2),point1(4),50,'k^','MarkerFaceColor','k');%每个区域的最低成本
scatter3(point2(1),point2(2),point2(4),50,'k^','MarkerFaceColor','k');
scatter3(point3(1),point3(2),point3(4),50,'k^','MarkerFaceColor','k');
scatter3(point4(1),point4(2),point4(4),50,'k^','MarkerFaceColor','k');
scatter3(point5(1),point5(2),point5(4),50,'k^','MarkerFaceColor','k');
scatter3(point6(1),point6(2),point6(4),50,'k^','MarkerFaceColor','k');
hold off;
xlabel('Pressure');
ylabel('Water-carbon ratio');
zlabel('Lowest cost');



data2=data(data(:,1)>=2850,:);%根据压力节点，压力大于2845的数据
figure(4);
scatter3(data2(:,1),data2(:,2),data2(:,4),'b.');
xlabel('Pressure');
ylabel('Water-carbon ratio');
zlabel('Lowest cost');
%%第一部分
data21=data2(data2(:,4)<250000,:);%为了区分块的颜色，选出成本小于250000的区域%%且是水碳比3.946165-5.079222
a=find(data21(:,4)==min(data21(:,4)));
point21=data21(a,:);
%%第二部分
data22=data2(data2(:,4)>250000&data2(:,4)<260000,:);%选出成本250000<*<260000的区域%且是水碳比3.203817-3.907094
a=find(data22(:,4)==min(data22(:,4)));
point22=data22(a,:);
%%第三部分
data31=data2(data2(:,2)>2.61&data2(:,2)<3.4&data2(:,4)>270000,:);%选出水碳比2.617753-3.399171
data32=data2(data2(:,1)<=3190&data2(:,2)<2.58&data2(:,2)>2.56,:);%选出水碳比为2.500540-2.578682，成本小于289100
data33=data2(data2(:,1)<=3050&data2(:,2)<2.54&data2(:,2)>2.53,:);
data34=data2(data2(:,1)<=2910&data2(:,2)<2.51&data2(:,2)>2.5,:);
data23=[data31;data32;data33;data34];
a=find(data23(:,4)==min(data23(:,4)));
point23=data23(a,:);
%%第四部分
data41=data2(data2(:,2)>2.34&data2(:,2)<2.47,:);%选出水碳比2.344256-2.461469
data42=data2(data2(:,1)>=3200&data2(:,2)<2.58&data2(:,2)>2.56,:);%选出水碳比为2.500540-2.578682，成本大于289100
data43=data2(data2(:,1)>=3060&data2(:,2)<2.54&data2(:,2)>2.3,:);
data44=data2(data2(:,1)>=2920&data2(:,2)<2.51&data2(:,2)>2.5,:);
data45=data2(data2(:,1)<=3150&data2(:,2)<2.31&data2(:,2)>2.3,:);
data46=data2(data2(:,1)<=2990&data2(:,2)<2.27&data2(:,2)>2.26,:);
data24=[data41;data42;data43;data44;data45;data46];
a=find(data24(:,4)==min(data24(:,4)));
point24=data24(a,:);
%%第五部分
data51=data2(data2(:,2)>2.1&data2(:,2)<2.23,:);%选出水碳比2.109830-2.227043
data52=data2(data2(:,1)>=3160&data2(:,2)<2.31&data2(:,2)>2.3,:);
data53=data2(data2(:,1)>=3000&data2(:,2)<2.27&data2(:,2)>2.26,:);
data54=data2(data2(:,1)<=3110&data2(:,2)<2.06&data2(:,2)>2.08,:);
data25=[data51;data52;data53;data54];
a=find(data25(:,4)==min(data25(:,4)));
point25=data25(a,:);
%%第六部分
data61=data2(data2(:,2)>1.95&data2(:,2)<2.04,:);%选出水碳比1.953547-2.031688
data62=data2(data2(:,1)>=3120&data2(:,2)<2.06&data2(:,2)>2.08,:);
data26=[data61;data62];
a=find(data26(:,4)==min(data26(:,4)));
point26=data26(a,:);

figure(5);
scatter3(data21(:,1),data21(:,2),data21(:,4),'b.');
hold on;
scatter3(data22(:,1),data22(:,2),data22(:,4),'g.');
scatter3(data23(:,1),data23(:,2),data23(:,4),'r.');
scatter3(data24(:,1),data24(:,2),data24(:,4),'c.');
scatter3(data25(:,1),data25(:,2),data25(:,4),'m.');
scatter3(data26(:,1),data26(:,2),data26(:,4),'k.');

scatter3(point21(1),point21(2),point21(4),50,'k^','MarkerFaceColor','k');%每个区域的最低成本
scatter3(point22(1),point22(2),point22(4),50,'k^','MarkerFaceColor','k');
scatter3(point23(1),point23(2),point23(4),50,'k^','MarkerFaceColor','k');
scatter3(point24(1),point24(2),point24(4),50,'k^','MarkerFaceColor','k');
scatter3(point25(1),point25(2),point25(4),50,'k^','MarkerFaceColor','k');
scatter3(point26(1),point26(2),point26(4),50,'k^','MarkerFaceColor','k');
hold off;
xlabel('Pressure');
ylabel('Water-carbon ratio');
zlabel('Lowest cost');
