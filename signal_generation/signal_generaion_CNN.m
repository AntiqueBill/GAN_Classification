close all;
clear all;
clc;

warning off

fc=70; %Carrier Frequency
fs=200;  %Sample Frequency
fd=2; %Code Rate
freqsep=1;  %Frequency Interval
N_code=40;  %Number of Symbols
length = 3000;%Final length of signals
N_samples_m = 20000;%Number of overlapped samples
N_samples_test = 2000;
num_classes = 6;

fc_max = 1.1;
fc_min = 0.9;

Ac_max = 1.1;
Ac_min = 0.9;

%for gg=-5:15
    
snr_max = 15;
snr_min = 0;
max_targets = 1;
min_targets = 1;

max_shift = fs*N_code/fd - length;

fprintf('Generating overlapped samples...\nMax_target = %d\n', max_targets);

train_x = zeros(N_samples_m, length);
train_y = zeros(N_samples_m, 1);

test_x = zeros(N_samples_test, length);
test_y = zeros(N_samples_test, 1);
%生成1×N_samples_m的min_targets: max_targets之间的均匀分布随机数
%决定每一行混几种信号
%idx_tar = randi([min_targets, max_targets], 1, N_samples_m);

for i=1:N_samples_m
    if mod(i, 2000) == 0
        fprintf('   itr=%d\n',i);
    end
    class_i = randperm(num_classes);%返回一行包含从1到num_classes（这里为8类）的整数。
    class_i = class_i(1);%决定混那几个信号
    fcc = unifrnd (fc_min, fc_max,1,1)*fc;%返回[size(class_i,2),1]大小的从fcmin到fcmax的随机数
    Acc = unifrnd (Ac_min, Ac_max,1,1);%如果为1个混合返回1个数，2个混合返回2个数
    shift = floor(unifrnd (1, max_shift));%相位偏移也需要随机
    y = zeros(1, length);%2种混合就是2行length列
    %for j =1:size(class_i,2)
        switch class_i
            case 1
                yr=ask2(N_code,fcc,fs,fd,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc;
                y= yr(shift:(shift+length-1));
                train_y(i, :)=1;
            case 2
                yr=fsk2(N_code,fcc,fs,fd,freqsep,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc;
                y= yr(shift:(shift+length-1));
                train_y(i, :)=2;
            case 3
                yr=psk2(N_code,fcc,fs,fd,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc;
                y= yr(shift:(shift+length-1));
                train_y(i, :)=3;
            case 4
                yr=psk4(N_code,fcc,fs,fd,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc;
                y= yr(shift:(shift+length-1));
                train_y(i, :)=4;
            case 5
                yr=qam16(N_code,fcc,fs,fd,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc;
                y= yr(shift:(shift+length-1));
                train_y(i, :)=5;
            case 6
                yr=msk(N_code,fs,fd,fcc,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc;
                y = yr(shift:(shift+length-1));
                train_y(i, :)=6;
        end
    %end
    y_r = mapminmax(y);
    snr = randi([snr_min, snr_max],1);
    train_x(i,:) = awgn(y_r, snr, 'measured','db');
end

for i=1:N_samples_test
    if mod(i, 2000) == 0
        fprintf('   itr=%d\n',i);
    end
    class_i = randperm(num_classes);%返回一行包含从1到num_classes（这里为8类）的整数。
    class_i = class_i(1);%决定混那几个信号
    fcc = unifrnd (fc_min, fc_max,size(class_i,2),1)*fc;%返回[size(class_i,2),1]大小的从fcmin到fcmax的随机数
    Acc = unifrnd (Ac_min, Ac_max,size(class_i,2),1);%如果为1个混合返回1个数，2个混合返回2个数
    shift = floor(unifrnd (1, max_shift));%相位偏移也需要随机
    y = zeros(1, length);%2种混合就是2行length列
    %for j =1:size(class_i,2)
        switch class_i
            case 1
                yr=ask2(N_code,fcc,fs,fd,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc;
                y = yr(shift:(shift+length-1));
                test_y(i, :)=1;
            case 2
                yr=fsk2(N_code,fcc,fs,fd,freqsep,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc;
                y = yr(shift:shift+length-1);
                test_y(i, :)=2;
            case 3
                yr=psk2(N_code,fcc,fs,fd,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc;
                y = yr(shift:shift+length-1);
                test_y(i, :)=3;
            case 4
                yr=psk4(N_code,fcc,fs,fd,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc;
                y = yr(shift:shift+length-1);
                test_y(i, :)=4;
            case 5
                yr=qam16(N_code,fcc,fs,fd,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc;
                y = yr(shift:shift+length-1);
                test_y(i, :)=5;
            case 6
                yr=msk(N_code,fs,fd,fcc,1);
                yr = yr/sqrt(sum(yr.^2)/(fs*N_code/fd))*Acc;
                y = yr(shift:shift+length-1);
                test_y(i, :)=6;
        end
    %end
    y_r = mapminmax(y);
    snr = randi([snr_min, snr_max],1);
    test_x(i,:) = awgn(y_r, snr, 'measured','db');
end

Ac = [Ac_min, Ac_max];
fc = [fc_min, fc_max];
snr = [snr_min, snr_max];
fprintf('Saving...\n');
train_x = train_x';
train_y = train_y';
test_x = test_x';
test_y = test_y';
save(strcat('../samples/dataset_CNN'),'train_x','train_y','test_x', 'test_y','Ac','fc', 'snr','length','-v7.3')
