
clear all;

begin_snr = -5;
end_snr = 10;

train_x1 = [];
train_y1 = [];
test_x1 = [];
test_y1 = [];


for snr =begin_snr:end_snr
    if snr <0
        fdata = strcat('dataset_MAMC','-',num2str(abs(snr)));
    else
        fdata = strcat('dataset_MAMC', num2str(snr));
    end
    
    load(strcat('../samples/',fdata,'_6.mat'))
    
    train_x1=[train_x1;x_train'];
    train_y1=[train_y1;y_train'];
   % test_x1=[test_x1;test_x];
  %  test_y1=[test_y1;test_y];
end


train_x = train_x1;
clear train_x1
train_y = train_y1;
clear train_y1
test_x = test_x1;
clear test_x1
test_y = test_y1;
clear test_y1


disp(strcat('Normalizing....'))
 
train_x=(train_x-mean(train_x(:)))/std(train_x(:));
test_x=(test_x-mean(test_x(:)))/std(test_x(:));



file_name = strcat('test',num2str(begin_snr),'_',num2str(end_snr));
tic
disp(strcat('start saving', 32,file_name,'.mat, please wait....'))
toc

tic

save(strcat('../samples/',file_name,'_20'),'train_x','train_y','-v7.3')

toc