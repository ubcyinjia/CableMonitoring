clear; clc;

datestr(now)

%specify the load model
load('synthetic_SNR_load1.mat');

%specify the subcarrier group
Test_Group=Group_9;

%specify the portion used for training
train_ratio=0.8;

%specify the arima model used
Mdl=arima(2,0,1);

train_index=1:1:floor(train_ratio*length(Time));
test_index=length(train_index)+1:1:length(Time);

EstMdl=estimate(Mdl,Test_Group');
y_estimate_Mdl=zeros(1,length(test_index));


for i=1:1:length(test_index)
    moving_X=Test_Group(length(train_index)+i-20:1:length(train_index)+i-1);
    y_estimate_Mdl(i)=forecast(EstMdl,1,moving_X');

end
y_measure=Test_Group(test_index);

%normalized RMSE for the baseline predictor
caliber=sqrt(var(diff(y_measure))/length(y_measure))

%normalized RMSE for the ARIMA model
rmse_Mdl=sqrt(var(y_estimate_Mdl-y_measure)/length(y_measure))


datestr(now)

%save('arima_test_G9.mat','y_estimate_Mdl7_0','y_measure','centers');