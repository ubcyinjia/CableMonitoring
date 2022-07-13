clear; close all; clc;

datestr(now)

%specify the load model
load('synthetic_SNR_load1.mat');

%specify the subcarrier group, Group_1 to Group_9
Test_Group=Group_2;

%specify the portion used for training
train_ratio=0.8;
train_index=1:1:floor(train_ratio*length(Time));
test_index=length(train_index)+1:1:length(Time);

%specify the window size
window_size=20;


%doing the data normalization
mu=mean(Test_Group);
std_exp=sqrt(var(Test_Group));
Group_Test_normal=(Test_Group-mu)/std_exp;

X_train=zeros(length(train_index)-window_size,window_size);
y_train=zeros(length(train_index)-window_size,1);

for i=1:1:length(train_index)-window_size
    X_train(i,:)=reshape(Group_Test_normal(i:1:i+window_size-1),1,[]);   
    y_train(i)=Group_Test_normal(i+window_size);

end

X_test=zeros(length(test_index)-window_size,window_size);
y_test=zeros(length(test_index)-window_size,1);

for i=1:1:length(test_index)-window_size
    X_test(i,:)=reshape(Group_Test_normal(...
        i+test_index(1)-1:1:i+window_size-1+test_index(1)-1),1,[]);
    
    y_test(i)=Group_Test_normal(i+window_size+test_index(1)-1);
    
end

csvwrite('X_train.csv',X_train);
csvwrite('X_test.csv',X_test);
csvwrite('y_train.csv',y_train);
csvwrite('y_test.csv',y_test);


