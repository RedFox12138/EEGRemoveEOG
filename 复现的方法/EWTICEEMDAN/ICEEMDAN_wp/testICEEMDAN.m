%% ICEEMDAN分解画图程序源码(公开版)
% 运行该代码前，请务必安装时频域分析工具箱TFA_Toolbox，获取地址：http://www.khscience.cn/docs/index.php/2020/04/09/1/
% 本程序2021.11定型，后续更新都在完整版代码中
%  知乎原文链接：https://zhuanlan.zhihu.com/p/434546183
%  本代码地址：http://www.khscience.cn/docs/index.php/2021/11/18/iceemdan/
%  完整版代码视频参考教程：http://www.khscience.cn/docs/index.php/2021/04/11/emdscourse/
%% 1.生成仿真信号
fs = 400;  %采样频率
t = 0:1/fs:0.75; %时间轴
x = sin(2*pi*4*t); %低频正弦信号
y = 0.5*sin(2*pi*120*t); %高频正弦信号
for i = 1:length(t) %将高频信号处理成间断性
    if mod(t(i),0.25)>0.11&&mod(t(i),0.25)<0.12
    else
        y(i) = 0;
    end
end
sig = x+y; %信号叠加
figure('color','white')
plot(t,sig,'k') %绘制原始信号
%% 2.ICEEMDAN分解图
Nstd = 0.2; %Nstd为附加噪声标准差与Y标准差之比
NE = 100;   %NE为对信号的平均次数
MaxIter = 1000;% MaxIter 最大迭代次数
imf = pICEEMDAN(sig,t,Nstd,NE,MaxIter);  %p文件可调用，无法查看源码，需要源码请获取完整版代码
% function imf = pICEEMDAN(data,FsOrT,Nstd,NE,MaxIter)
% 画信号ICEEMDAN分解图
% 输入：
% data为待分解信号
% FsOrT为采样频率或采样时间向量，如果为采样频率，该变量输入单个值；如果为时间向量，该变量为与y相同长度的一维向量。如果未知采样频率，可设置为1
% Nstd为附加噪声标准差与Y标准差之比
% NE为对信号的平均次数
% MaxIter：最大迭代次数
% 输出：
% imf为经ICEEMDAN分解后的各imf分量值
% 例1：（FsOrT为采样频率）
% fs = 100;
% t = 1/fs:1/fs:1;
% data = sin(2*pi*5*t)+2*sin(2*pi*20*t);
% imf = pICEEMDAN(data,fs,0.2,100);
% 例2：（FsOrT为时间向量，需要注意此时FsOrT的长度要与y相同）
% t = 0:0.01:1;
% data = sin(2*pi*5*t)+2*sin(2*pi*20*t);
% imf = pICEEMDAN(data,t,0.2,100);
%% 3.ICEEMDAN分解及频谱图
imf = pICEEMDANandFFT(sig,t,Nstd,NE,MaxIter);  %p文件可调用，无法查看源码，需要源码请获取完整版代码
% function imf = pICEEMDANandFFT(y,FsOrT,Nstd,NE,MaxIter)
% 画信号ICEEMDAN分解与各IMF分量频谱对照图
% 输入：
% y为待分解信号
% FsOrT为采样频率或采样时间向量，如果为采样频率，该变量输入单个值；如果为时间向量，该变量为与y相同长度的一维向量
% Nstd为附加噪声标准差与Y标准差之比
% NE为对信号的平均次数
% MaxIter：最大迭代次数
% 输出：
% imf为经ICEEMDAN分解后的各imf分量值
% 例1：（FsOrT为采样频率）
% fs = 100;
% t = 1/fs:1/fs:1;
% y = sin(2*pi*5*t)+2*sin(2*pi*20*t);
% imf = pICEEMDANandFFT(y,fs,0.2,100);
% 例2：（FsOrT为时间向量，需要注意此时FsOrT的长度要与y相同）
% t = 0:0.01:1;
% y = sin(2*pi*5*t)+2*sin(2*pi*20*t);
% imf = pICEEMDANandFFT(y,t,0.2,100);
%% 关于完整版代码：
% 公开版代码的函数文件为p文件，可以被调用，但无法查看代码。
% 完整版代码中全部为m文件，m文件可以查看源码并自由修改。
% 如果需要封装好的画图函数（pICEEMDAN.m、pICEEMDANandFFT.m、kICEEMDAN.m和pFFT.m）的源码，可在下述连接（完整版）获取：
% http://www.khscience.cn/docs/index.php/2021/11/18/iceemdan/