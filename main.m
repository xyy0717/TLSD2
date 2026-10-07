clc;
clear;

addpath(genpath(pwd));
addpath 'D:\mine-HP\mazing6\PML';

foldpath = 'D:\mine-HP\mazing6\PML';
allContents = dir(foldpath);

%% ========================================================================
%% 全局配置
%% ========================================================================

config.nfold = 5;
config.seed = 42;

config.mlknn_k = 10;
config.mlknn_smooth = 1;

config.eps0 = 1e-12;

% ===== 四个参数统一搜索范围 =====
config.search_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000];

% ===== 特征选择比例 =====
config.feature_threshold_map = [...
    100,    0.4;
    500,    0.3;
    1000,   0.2;
    Inf,    0.1];

%% ========================================================================
%% 主循环
%% ========================================================================

for iu = 3:length(allContents)

    dataset_name = allContents(iu).name;
    dataset_name_short = dataset_name(1:end-4);

    fprintf('\n%s\n', repmat('=', 1, 80));
    fprintf('TLSD2 网格搜索: %s\n', dataset_name_short);
    fprintf('%s\n', repmat('=', 1, 80));

    try

        %% =================================================================
        %% 1. 加载并预处理数据
        %% =================================================================

        load(fullfile(foldpath, dataset_name));

        data = columnMinMaxNormalization(data);

        [N, num_feature] = size(data);

        fprintf('样本数: %d\n', N);
        fprintf('特征数: %d\n', num_feature);


        %% =================================================================
        %% 2. 确定特征选择数量
        %% =================================================================

        Theta = 0.1;

        for ti = 1:size(config.feature_threshold_map, 1)

            if num_feature <= config.feature_threshold_map(ti, 1)

                Theta = config.feature_threshold_map(ti, 2);
                break;

            end

        end

        k_select = max(1, fix(Theta * num_feature));

        fprintf('Theta = %.2f\n', Theta);
        fprintf('选择特征数 = %d\n\n', k_select);


        %% =================================================================
        %% 3. 固定五折划分
        %% =================================================================

        rng(config.seed);

        indices_all = crossvalind(...
            'Kfold', ...
            1:N, ...
            config.nfold);


        %% =================================================================
        %% 4. 提前计算每一折中与参数无关的数据
        %% =================================================================

        fold_data = cell(config.nfold, 1);

        for i = 1:config.nfold

            test_idxs = (indices_all == i);
            train_idxs = ~test_idxs;

            % ===== 特征 =====
            train_X = data(train_idxs, :);
            test_X  = data(test_idxs, :);

            % ===== 候选标签 =====
            tr_target = candidate_labels(:, train_idxs);

            % TLSD2 使用的标签
            train_Y_pmfs = tr_target;
            train_Y_pmfs(train_Y_pmfs == -1) = 0;
            train_Y_pmfs = train_Y_pmfs';

            % ML-KNN 训练标签
            train_Y_ml = tr_target;
            train_Y_ml(train_Y_ml == 0) = -1;

            % ML-KNN 测试标签
            test_Y_ml = target(:, test_idxs);
            test_Y_ml(test_Y_ml == 0) = -1;


            %% ===== 计算 R =====

            Xc = train_X - mean(train_X, 1);

            colNorm = sqrt(sum(Xc.^2, 1)) + config.eps0;

            Z = bsxfun(@rdivide, Xc, colNorm);

            C = Z' * Z;

            R_temp = C .* C;

            R = 0.5 * (R_temp + R_temp');


            %% ===== 保存当前折数据 =====

            fold_data{i}.train_X = train_X;
            fold_data{i}.test_X = test_X;

            fold_data{i}.train_Y_pmfs = train_Y_pmfs;
            fold_data{i}.train_Y_ml = train_Y_ml;
            fold_data{i}.test_Y_ml = test_Y_ml;

            fold_data{i}.R = R;

        end


        %% =================================================================
        %% 5. 网格搜索初始化
        %% =================================================================

        search_list = config.search_list;

        n_param = length(search_list);

        total_combinations = n_param^4;

        fprintf('参数搜索范围:\n');
        fprintf('alpha  = [0.001 0.01 0.1 1 10 100 1000]\n');
        fprintf('beta   = [0.001 0.01 0.1 1 10 100 1000]\n');
        fprintf('gamma  = [0.001 0.01 0.1 1 10 100 1000]\n');
        fprintf('lambda = [0.001 0.01 0.1 1 10 100 1000]\n');

        fprintf('\n总参数组合数 = %d\n\n', total_combinations);


        %% ===== 最优结果 =====

        best_AP = -inf;

        best_alpha = NaN;
        best_beta = NaN;
        best_gamma = NaN;
        best_lambda = NaN;

        best_mean_metrics = [];
        best_std_metrics = [];

        combination_id = 0;

        total_start = tic;


        %% =================================================================
        %% 6. 四参数完整网格搜索
        %% =================================================================

        for ia = 1:n_param

            alpha = search_list(ia);

            for ib = 1:n_param

                beta = search_list(ib);

                for ig = 1:n_param

                    gamma = search_list(ig);

                    for il = 1:n_param

                        lambda = search_list(il);

                        combination_id = combination_id + 1;


                        %% ===== TLSD2 参数 =====

                        current_params = struct();

                        current_params.alpha = alpha;
                        current_params.beta = beta;
                        current_params.gamma = gamma;
                        current_params.lambda = lambda;

                        current_params.verbose = false;


                        %% =================================================
                        %% 当前参数组合五折验证
                        %% =================================================

                        cv_results = zeros(config.nfold, 7);

                        for i = 1:config.nfold

                            train_X = fold_data{i}.train_X;
                            test_X = fold_data{i}.test_X;

                            train_Y_pmfs = fold_data{i}.train_Y_pmfs;

                            train_Y_ml = fold_data{i}.train_Y_ml;
                            test_Y_ml = fold_data{i}.test_Y_ml;

                            R = fold_data{i}.R;


                            %% ===== TLSD2 =====

                            [W, G, ~, a, ~, ~] = TLSD2(...
                                train_X, ...
                                train_Y_pmfs, ...
                                R, ...
                                current_params);


                            %% ===== 特征评分 =====

                            G_colNorm = sqrt(sum(G.^2, 1))';

                            W_rowNorm = sqrt(sum(W.^2, 2));

                            feature_scores = ...
                                a .* G_colNorm .* W_rowNorm;


                            %% ===== 选择特征 =====

                            [~, sorted_idx] = ...
                                sort(feature_scores, 'descend');

                            selected_features = ...
                                sorted_idx(1:k_select);


                            %% ===== ML-KNN =====

                            [Prior, PriorN, Cond, CondN] = ...
                                MLKNN_train(...
                                train_X(:, selected_features), ...
                                train_Y_ml, ...
                                config.mlknn_k, ...
                                config.mlknn_smooth);


                            [HL, RL, Cov, AP, macf1, micf1, OE, ~, ~] = ...
                                MLKNN_test(...
                                train_X(:, selected_features), ...
                                train_Y_ml, ...
                                test_X(:, selected_features), ...
                                test_Y_ml, ...
                                config.mlknn_k, ...
                                Prior, ...
                                PriorN, ...
                                Cond, ...
                                CondN);


                            %% ===== 当前折结果 =====

                            cv_results(i, :) = ...
                                [HL, RL, OE, AP, macf1, micf1, Cov];

                        end


                        %% =================================================
                        %% 五折均值和标准差
                        %% =================================================

                        mean_metrics = mean(cv_results, 1);
                        std_metrics = std(cv_results, 0, 1);

                        current_AP = mean_metrics(4);


                        %% =================================================
                        %% 更新最优参数
                        %% 目前以 Average Precision 最大作为标准
                        %% =================================================

                        if current_AP > best_AP

                            best_AP = current_AP;

                            best_alpha = alpha;
                            best_beta = beta;
                            best_gamma = gamma;
                            best_lambda = lambda;

                            best_mean_metrics = mean_metrics;
                            best_std_metrics = std_metrics;


                            fprintf(['New Best [%d/%d] ', ...
                                'alpha=%.3g, beta=%.3g, ', ...
                                'gamma=%.3g, lambda=%.3g, ', ...
                                'AP=%.6f\n'], ...
                                combination_id, ...
                                total_combinations, ...
                                alpha, ...
                                beta, ...
                                gamma, ...
                                lambda, ...
                                current_AP);

                        end


                        %% ===== 每 100 组输出一次进度 =====

                        if mod(combination_id, 100) == 0

                            fprintf(...
                                'Progress: %d/%d, current best AP = %.6f\n', ...
                                combination_id, ...
                                total_combinations, ...
                                best_AP);

                        end

                    end
                end
            end
        end


        %% =================================================================
        %% 7. 输出最终最优结果
        %% =================================================================

        elapsed = toc(total_start);

        fprintf('\n%s\n', repmat('=', 1, 80));
        fprintf('最优参数搜索完成\n');
        fprintf('%s\n', repmat('=', 1, 80));


        fprintf('\nBest Parameters:\n');

        fprintf('alpha  = %.6g\n', best_alpha);
        fprintf('beta   = %.6g\n', best_beta);
        fprintf('gamma  = %.6g\n', best_gamma);
        fprintf('lambda = %.6g\n\n', best_lambda);


        fprintf('Five-fold Results:\n');

        fprintf('Hamming Loss      = %.6f ± %.6f\n', ...
            best_mean_metrics(1), best_std_metrics(1));

        fprintf('Ranking Loss      = %.6f ± %.6f\n', ...
            best_mean_metrics(2), best_std_metrics(2));

        fprintf('One Error         = %.6f ± %.6f\n', ...
            best_mean_metrics(3), best_std_metrics(3));

        fprintf('Average Precision = %.6f ± %.6f\n', ...
            best_mean_metrics(4), best_std_metrics(4));

        fprintf('Macro-F1          = %.6f ± %.6f\n', ...
            best_mean_metrics(5), best_std_metrics(5));

        fprintf('Micro-F1          = %.6f ± %.6f\n', ...
            best_mean_metrics(6), best_std_metrics(6));

        fprintf('Coverage           = %.6f ± %.6f\n', ...
            best_mean_metrics(7), best_std_metrics(7));


        fprintf('\n总参数组合: %d\n', total_combinations);
        fprintf('总耗时: %.2f 秒\n', elapsed);


        fprintf('\n最优组合：\n');
        fprintf(...
            'alpha=%.6g, beta=%.6g, gamma=%.6g, lambda=%.6g\n', ...
            best_alpha, ...
            best_beta, ...
            best_gamma, ...
            best_lambda);


    catch ME

        fprintf('\n处理数据集 %s 时出错:\n', dataset_name_short);

        fprintf('%s\n', ME.message);

        fprintf('错误堆栈:\n%s\n', getReport(ME));

    end
end


fprintf('\n%s\n', repmat('=', 1, 80));
fprintf('TLSD2 grid search finished!\n');
fprintf('%s\n', repmat('=', 1, 80));