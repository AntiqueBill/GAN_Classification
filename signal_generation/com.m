clc
clear all

begin_snr = -10;
end_snr = 5;

train_x1 = [];
train_y1 = [];
x_pure1 = [];
x_simple1 = [];


for snr =begin_snr:2:end_snr
    if snr <0
        fdata = strcat('dataset_MAMC','-',num2str(abs(snr)),'_6');
    else
        fdata = strcat('dataset_MAMC', num2str(snr),'_6');
    end
    
    load(strcat('../samples/',fdata,'.mat'))
    
    train_x1=[train_x1,x_train];
    train_y1=[train_y1,y_train];
    x_pure1=[x_pure1,x_pure];
    x_simple1 = [x_simple1,x_simple];
end


train_x = train_x1;
clear train_x1
train_y = train_y1;
clear train_y1
x_pure = x_pure1;
clear x_pure1
x_simple = x_simple1;
clear x_simple1


disp(strcat('Normalizing....'))
 %train_x=(train_x-mean(train_x(:)))/std(test_x(:));
%test_x=(test_x-mean(test_x(:)))/std(test_x(:));



file_name = strcat('../samples/test',num2str(begin_snr),'_',num2str(end_snr));
tic
disp(strcat('start saving', 32,file_name,'.mat, please wait....'))
toc

tic

save(strcat('../samples/',file_name),'train_x','train_y','x_pure','x_simple','-v7.3')

toc