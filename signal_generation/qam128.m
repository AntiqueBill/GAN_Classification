function y=qam128(N_code,fc,fs,fd,Ac)
    M=128;
    k=log2(M);
    n=N_code*k;%比特序列长度 
    N_samples=fs/fd;%过采样率 
    x=randi([0,1],1,n);%生成随机二进制比特流
    %figure(1)
    %stem(x(1:n),'filled');%画出相应的二进制比特流信号 
    %title('二进制随机比特流');
    %xlabel('比特序列');
    %ylabel('信号幅度');
    x4=reshape(x,k,length(x)/k);%将原始的二进制比特序列每四个一组分组，并排列成k行length(x)/k列的矩阵 
    xsym=bi2de(x4.','left-msb');%将矩阵转化为相应的64进制信号序列 
   % figure(2);
    %stem(xsym);%画出相应的16进制信号序列 
    %title('64进制随机信号');
    %xlabel('信号序列');
   % ylabel('信号幅度');
    y=modulate(modem.qammod(M),xsym);%用64QAM调制器对信号进行 
    
    
    t=1:N_samples;
    carrier=Ac*cos(2*pi*fc*t/fs);%产生信号脉冲g(t) 
    gt=ones(1,length(carrier));
    %生成调制信号S(t) 
    St_complex=zeros(1,length(carrier)*length(y));
    for n1=1:length(y)
        St_complex((N_samples*(n1-1)+1):(N_samples*(n1-1)+N_samples))=(y(n1)*carrier).*gt;
    end
  %  figure(3)
    y=real(St_complex);
   % plot(St_real);
    %title('QAM仿真波形图 g(t)为升余弦脉冲');
    %xlabel('采样点')
    %ylabel('幅度')