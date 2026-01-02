% split_eeg_segments.m
% 交互式脚本：读取一维 EEG mat 文件，用户输入区间，分割为脑电、眨眼、眼动三类数据并保存

% 1. 选择原始数据文件
[file, path] = uigetfile('*.mat', '选择一维 EEG 数据文件');
if isequal(file,0)
    disp('未选择文件，退出'); return;
end
matfile = fullfile(path, file);
S = load(matfile);
varnames = fieldnames(S);
data = [];
for k=1:numel(varnames)
    v = S.(varnames{k});
    if isnumeric(v) && isvector(v)
        data = double(v(:)); break;
    end
end
if isempty(data)
    error('未找到一维数值变量');
end
N = numel(data);
fprintf('数据长度: %d\n', N);

% 2. 可视化数据，辅助区间选择
figure; plot(data); title('原始EEG数据'); xlabel('样本点'); ylabel('幅值');
disp('请在命令行输入区间索引（如 [1000 2000]），可多段。');

% 3. 输入区间
brain_idx = input('输入脑电区间（如 [1000 2000; 3000 4000]）：');
wink_idx  = input('输入眨眼区间（如 [5000 5200; 8000 8200]）：');
eye_idx   = input('输入眼动区间（如 [12000 12500]）：');

% 4. 分割并保存
cd(path); % 保存到原始数据同目录
save_segments(brain_idx, data, 'Brain');
save_segments(wink_idx,  data, 'Wink');
save_segments(eye_idx,   data, 'EyeMove');

disp('所有分段已保存，可用于 SNR 计算脚本。');

function save_segments(idx, data, prefix)
    if isempty(idx)
        return;
    end
    for i=1:size(idx,1)
        seg = data(idx(i,1):idx(i,2));
        fname = sprintf('%s_%d.mat', prefix, i);
        save(fname, 'seg');
        fprintf('已保存 %s [%d:%d]\n', fname, idx(i,1), idx(i,2));
    end
end