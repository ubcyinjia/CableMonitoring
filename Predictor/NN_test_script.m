    y_predict=csvread('snr_prediction.csv');

test_length=length(y_test);
coeff_2=sqrt(var(Test_Group));

y_predict=y_predict*std_exp+mu;
y_measure=y_test(1:1:test_length)*std_exp+mu;

%normalized RMSE for the predictor
rmse=sqrt(var(y_predict-y_measure))/coeff_2

%normalized RMSE for the baseline
caliber=sqrt(var(diff(y_measure)))/coeff_2

