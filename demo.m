clc;
clear;

addpath(genpath(pwd));

%% ========================================================================
%% 1. 直接加载数据集
%% ========================================================================
load('music_style.mat');

% 数据要求：
% data             : N × d
% candidate_labels : L × N
% target           : L × N

data = columnMinMaxNormalization(data);

[N, num_feature] = size(data);

fprintf('\n============================================================\n');
fprintf('Dataset: music_style\n');
fprintf('Samples : %d\n', N);
fprintf('Features: %d\n', num_feature);
fprintf('Labels  : %d\n', size(candidate_labels, 1));
fprintf('============================================================\n\n');


%% ========================================================================
%% 2. 直接填写最优参数
%% ========================================================================
% 在这里修改成你的最优参数

alpha  = 100;
beta   = 1;
gamma  = 0.001;
lambda = 0.001;

fprintf('Optimal parameters:\n');
fprintf('alpha  = %.6g\n', alpha);
fprintf('beta   = %.6g\n', beta);
fprintf('gamma  = %.6g\n', gamma);
fprintf('lambda = %.6g\n\n', lambda);


%% ========================================================================
%% 3. 基本配置
%% ========================================================================
nfold = 5;

mlknn_k = 10;
mlknn_smooth = 1;

eps0 = 1e-12;

% 如果希望每次运行使用相同的五折划分，则保留
rng(42);


%% ========================================================================
%% 4. 确定选择特征数量
%% ========================================================================
if num_feature <= 100

    Theta = 0.4;

elseif num_feature <= 500

    Theta = 0.3;

elseif num_feature <= 1000

    Theta = 0.2;

else

    Theta = 0.1;

end

k_select = max(1, fix(Theta * num_feature));

fprintf('Theta = %.2f\n', Theta);
fprintf('Selected features = %d\n\n', k_select);


%% ========================================================================
%% 5. 五折交叉验证
%% ========================================================================
indices_all = crossvalind('Kfold', 1:N, nfold);

% 指标顺序：
% HL, RL, OE, AP, MacroF1, MicroF1, Coverage

cv_results = zeros(nfold, 7);


%% ========================================================================
%% 6. TLSD2 参数
%% ========================================================================
opts = struct();

opts.alpha   = alpha;
opts.beta    = beta;
opts.gamma   = gamma;
opts.lambda  = lambda;
opts.verbose = false;


%% ========================================================================
%% 7. 开始五折实验
%% ========================================================================
t_start = tic;

for i = 1:nfold

    fprintf('================ Fold %d / %d ================\n', ...
        i, nfold);

    %% ------------------------------------------------------------
    % 划分训练集和测试集
    %% ------------------------------------------------------------
    test_idxs  = (indices_all == i);
    train_idxs = ~test_idxs;

    train_X = data(train_idxs, :);
    test_X  = data(test_idxs, :);


    %% ------------------------------------------------------------
    % Partial Multi-Label 标签
    %% ------------------------------------------------------------
    tr_target = candidate_labels(:, train_idxs);

    train_Y_pmfs = tr_target;

    train_Y_pmfs(train_Y_pmfs == -1) = 0;

    train_Y_pmfs = train_Y_pmfs';


    %% ------------------------------------------------------------
    % ML-KNN 标签
    %% ------------------------------------------------------------
    train_Y_ml = tr_target;

    train_Y_ml(train_Y_ml == 0) = -1;

    test_Y_ml = target(:, test_idxs);

    test_Y_ml(test_Y_ml == 0) = -1;


    %% ------------------------------------------------------------
    % 计算特征冗余矩阵 R
    %% ------------------------------------------------------------
    Xc = train_X - mean(train_X, 1);

    colNorm = sqrt(sum(Xc.^2, 1)) + eps0;

    Z = bsxfun(@rdivide, Xc, colNorm);

    C = Z' * Z;

    R_temp = C .* C;

    R = 0.5 * (R_temp + R_temp');


    %% ------------------------------------------------------------
    % TLSD2
    %% ------------------------------------------------------------
    [W, G, ~, a, ~, ~] = TLSD2( ...
        train_X, ...
        train_Y_pmfs, ...
        R, ...
        opts);


    %% ------------------------------------------------------------
    % 特征评分
    %% ------------------------------------------------------------
    G_colNorm = sqrt(sum(G.^2, 1))';

    W_rowNorm = sqrt(sum(W.^2, 2));

    feature_scores = a .* G_colNorm .* W_rowNorm;


    %% ------------------------------------------------------------
    % 选择前 k 个特征
    %% ------------------------------------------------------------
    [~, sorted_idx] = sort(feature_scores, 'descend');

    selected_features = sorted_idx(1:k_select);


    %% ------------------------------------------------------------
    % ML-KNN 训练
    %% ------------------------------------------------------------
    [Prior, PriorN, Cond, CondN] = MLKNN_train( ...
        train_X(:, selected_features), ...
        train_Y_ml, ...
        mlknn_k, ...
        mlknn_smooth);


    %% ------------------------------------------------------------
    % ML-KNN 测试
    %% ------------------------------------------------------------
    [HL, RL, Cov, AP, macf1, micf1, OE, ~, ~] = MLKNN_test( ...
        train_X(:, selected_features), ...
        train_Y_ml, ...
        test_X(:, selected_features), ...
        test_Y_ml, ...
        mlknn_k, ...
        Prior, PriorN, Cond, CondN);


    %% ------------------------------------------------------------
    % 保存本折结果
    %% ------------------------------------------------------------
    cv_results(i, :) = ...
        [HL, RL, OE, AP, macf1, micf1, Cov];


    fprintf('HL      = %.6f\n', HL);
    fprintf('RL      = %.6f\n', RL);
    fprintf('OE      = %.6f\n', OE);
    fprintf('AP      = %.6f\n', AP);
    fprintf('MacroF1 = %.6f\n', macf1);
    fprintf('MicroF1 = %.6f\n', micf1);
    fprintf('Coverage= %.6f\n\n', Cov);

end


%% ========================================================================
%% 8. 计算最终性能
%% ========================================================================
mean_metrics = mean(cv_results, 1);

std_metrics = std(cv_results, 0, 1);

elapsed = toc(t_start);


%% ========================================================================
%% 9. 输出最终结果
%% ========================================================================
metric_names = { ...
    'HammingLoss', ...
    'RankingLoss', ...
    'OneError', ...
    'AveragePrecision', ...
    'MacroF1', ...
    'MicroF1', ...
    'Coverage'};


fprintf('\n\n');
fprintf('============================================================\n');
fprintf('              Final Performance of TLSD2\n');
fprintf('============================================================\n');

for k = 1:length(metric_names)

    fprintf('%-18s = %.3f +/- %.3f\n', ...
        metric_names{k}, ...
        mean_metrics(k), ...
        std_metrics(k));

end

fprintf('------------------------------------------------------------\n');

fprintf('Running time       = %.2f seconds\n', elapsed);

fprintf('------------------------------------------------------------\n');

fprintf('alpha  = %.6g\n', alpha);
fprintf('beta   = %.6g\n', beta);
fprintf('gamma  = %.6g\n', gamma);
fprintf('lambda = %.6g\n', lambda);

fprintf('============================================================\n');