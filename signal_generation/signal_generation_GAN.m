close all;
clear all;
clc;

warning off

fc=3.5; %Carrier Frequency
fs=2;  %Sample Frequency
fd=0.1; %Code Rate
freqsep=0.15;  %Frequency Interval
N_code=200;  %Number of Symbols
length = 3000;%Final length of signals
N_samples_m = 60000;%Number of overlapped samples
N_samples_test = 6000;%Number of overlapped samples
num_classes = 6;

fc_max = 1.1;
fc_min = 0.9;

Ac_max = 1.1;
Ac_min = 0.9;

%for gg=-20:5
    
snr_max = gg;
snr_min = gg;
max_targets = 2;
min_targets = 2;

max_shift = fs*N_code/fd - length;

fprintf('Generating overlapped samples...\nMax_target = %d\n', max_targets);

train_x = zeros(N_samples_m, length);
x_simple1= zeros(N_samples_m, length);
x_simple2= zeros(N_samples_m, length);
y_simple1= zeros(N_samples_m, 1);
y_simple2= zeros(N_samples_m, 1);
x_pure =  zeros(N_samples_m, length);
train_y = zeros(N_samples_m, num_classes);

test_x = zeros(N_samples_test, length);
test_x_simple1= zeros(N_samples_test, length);
test_x_simple2= zeros(N_samples_test, length);
test_y_simple1= zeros(N_samples_test, 1);
test_y_simple2= zeros(N_samples_test, 1);
test_x_pure =  zeros(N_samples_test, length);
test_y = zeros(N_samples_test, num_classes);
%生成1×N_samples_m的min_targets: max_targets之间的均匀分布随机数
%决定每一行混几种信号
idx_tar = randi([min_targets, max_targets], 1, N_samples_m);

for i=1:N_samples_m
    if mod(i, 2000) == 0
        fprintf('   itr=%d\n',i);
    end
    class_i = randperm(num_classes);%返回一行包含从1到num_classes（这里为8类）的整数。
    class_i = class_i(1:idx_tar(i));%决定混那几个信号
    class_i = sort(class_i);
    fcc = unifrnd (fc_min, fc_max,size(class_i,2),1);%返回[size(class_i,2),1]大小的从fcmin到fcmax的随机数
    Acc = unifrnd (Ac_min, Ac_max,size(class_i,2),1);%如果为1个混合返回1个数，2个混合返回2个数
    shift = unifrnd (0, max_shift,size(class_i,2),1);%相位偏移也需要随机
    y = zeros(idx_tar(i), length);%2种混合就是2行length列
    for j =1:size(class_i,2)
        switch class_i(j)
            case 1
                yr=ask2(N_code,fcc(j),fs,fd,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc(j);
                y(j,:) = yr(1, shift(j):shift(j)+length-1);
                train_y(i, class_i(j))=1;
            case 2
                yr=fsk2(N_code,fcc(j),fs,fd,freqsep,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc(j);
                y(j,:) = yr(1, shift(j):(shift(j)+length-1));
                train_y(i, class_i(j))=1;
%             case 3
%                 yr=fsk4(N_code,fcc(j),fs,fd,freqsep,1);
%                 yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc(j);
%                 y(j,:) = yr(1, shift(j):shift(j)+length-1);
%                 y_train(i, class_i(j))=1;
            case 3
                yr=psk2(N_code,fcc(j),fs,fd,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc(j);
                y(j,:) = yr(1, shift(j):(shift(j)+length-1));
                train_y(i, class_i(j))=1;
            case 4
                yr=psk4(N_code,fcc(j),fs,fd,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc(j);
                y(j,:) = yr(1, shift(j):shift(j)+length-1);
                train_y(i, class_i(j))=1;
            case 5
                yr=qam16(N_code,fcc(j),fs,fd,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc(j);
                y(j,:) = yr(1, shift(j):shift(j)+length-1);
                train_y(i, class_i(j))=1;
%             case 7
%                 yr=qam64(N_code,fcc(j),fs,fd,1);
%                 yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc(j);
%                 y(j,:) = yr(1, shift(j):shift(j)+length-1);
%                 y_train(i, class_i(j))=1;
            case 6
                yr=msk(N_code,fs,fd,fcc(j),1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc(j);
                y(j,:) = yr(1, shift(j):(shift(j)+length-1));
                train_y(i, class_i(j))=1;
        end
    end
    x_simple((2*i-1):2*i, :) = mapminmax(y,0,1);
    y_r = mapminmax(sum(y, 1),0,1);
    x_pure(i,:) = y_r;
    snr = randi([snr_min, snr_max],1);
    train_x(i,:) = awgn(y_r, snr, 'measured','db');
end

Ac = [Ac_min, Ac_max];
fc = [fc_min, fc_max];
snr = [snr_min, snr_max];
fprintf('Saving...\n');
x_simple1 = x_simple(1:2:(220000-1),:);
x_simple2 = x_simple(2:2:220000,:);
train_x = train_x';
x_simple = x_simple';
x_simple1 = x_simple1';
x_simple2 = x_simple2';
x_pure = x_pure';
train_y = train_y';
save(strcat('../samples/dataset_MAMC', num2str(gg), '_',num2str(num_classes)),'x_train','y_train','x_pure','x_simple','Ac','fc', 'snr','length','-v7.3')
%end