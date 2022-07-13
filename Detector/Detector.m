clear;close all;

delta_error_train=zeros(9,50941);
delta_error_test=zeros(9,12741);

for i=1:1:9
    %choose which set of files to load    
    %load model 1, predcited using ARIMA(2,0,1), medium distributed fault
    file_name=sprintf('DF_L1_T1_G%d_ARIMA7_distribution_medium',i);
    %load model 1, predcited using ARIMA(2,0,1), slight distributed fault
    %file_name=sprintf('DF_L1_T1_G%d_ARIMA7_distribution',i);
    
    load(file_name);
    delta_error_train(i,:)=y_estimate_Mdl7_1_train-y_measure_train;
    delta_error_test(i,:)=y_estimate_Mdl7_1-y_measure; 
end

cov_matrix=cov(delta_error_train');
inv_conv_matrix=inv(cov_matrix);
B=diag(delta_error_test'*inv(cov_matrix)*delta_error_test);


result_index=1:1:length(B);
figure;
plot(result_index/96,B);
xlabel('Days');
ylabel('Squared Mahalanobis Distance');
grid on;
