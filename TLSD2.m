function [W, G, P, a, obj, R] = TLSD2(X, Y, R, opts)
% TLSD2  Projected/proximal optimization for the TLSD^2 objective
% 规模归一化优化模型:
%   min_{G,W,P,a}
%     (1/(nL))||(X∘G)W - P||_F^2
%   + α*(1/n) Σ_i log(1+exp(v_i-u_i))
%   + β*(1/d) Σ_j ( (mean_i G_ij)^2 / (a_j+ε) )
%   + β*(1/d) Σ_j ( (||W_{j:}||_2/sqrt(L)) / (a_j+ε) )
%   + γ*(1/n) Σ_i ( (1-||P_{i:}||_2^2) / (1-1/|C_i|) )
%   + λ*( a' R a / (sum(R(:))+ε) )
%
% s.t. 0≤G≤1, 0≤P≤Y, Σ_l P_il=1, 0<a≤1
%
% P更新: CCCP/MM (与归一化目标一致)
%   Q_i = S_i + (γ*L*w_i)*P_old_i ,  w_i = 1/(1-1/|C_i|), |C_i|=1 -> w_i=0
%   P <- Proj_C(Q)
%
% W/G/a 更新: 自适应步长的投影/近端梯度更新
%
% 输入:
%   X - 特征矩阵 [n×d]
%   Y - 标签矩阵 [n×L]
%   R - 特征冗余矩阵 [d×d] (前置输入，若为空则自动计算)
%   opts - 参数结构体
%          必需字段: alpha, beta, gamma, lambda
%          其中 alpha 对应候选/非候选标签对比损失，beta 同时控制 G 与 W 的特征稀疏项
%
% 输出:
%   W, G, P, a - 优化变量
%   obj - 目标函数值序列
%   R - 特征冗余矩阵

    % ==================== 参数检查与设置 ====================
    if nargin < 4 || isempty(opts)
        opts = struct();
    end

    req_fields = {'alpha','beta','gamma','lambda'};
    for k = 1:numel(req_fields)
        if ~isfield(opts, req_fields{k})
            error('opts.%s 是必需的.', req_fields{k});
        end
    end

    opts = setDefaultOpts(opts);

    alpha  = opts.alpha;
    beta   = opts.beta;
    gamma  = opts.gamma;
    lambda = opts.lambda;
    tau    = opts.tau;
    eps0   = opts.eps;

    rng(opts.seed, 'twister');

    % ==================== 尺寸与初始化 ====================
    [n, d] = size(X);
    L = size(Y, 2);
    Ymask = (Y > 0);

    if opts.verbose
        fprintf('======== TLSD2 (投影/近端梯度版本) 初始化 ========\n');
        fprintf('数据维度: n=%d, d=%d, L=%d\n', n, d, L);
    end

    % 稀疏存储Ymask
    if nnz(Ymask) / numel(Ymask) < 0.3
        Ymask = sparse(Ymask);
    end

    % 处理特征冗余矩阵 R
    if nargin < 3 || isempty(R)
        if opts.verbose
            fprintf('未提供R矩阵，正在计算...\n');
        end
        R = computeR_cos2_psd(X, eps0);
    else
        % 验证R的有效性
        if size(R,1) ~= d || size(R,2) ~= d
            error('R矩阵维度不匹配: 期望[%d×%d], 实际[%d×%d]', d, d, size(R,1), size(R,2));
        end
        % 确保对称性
        R = 0.5 * (R + R');
    end
    sumR = sum(R(:)) + eps0;

    % 处理候选集（包括空候选集）
    [candIdx, noncandIdx, ~, emptyMask] = getCandidateIndices(Ymask, n, L);

    % 向量化权重计算
    wVec = computeWeights_vectorized(candIdx, n, emptyMask);

    % 初始化变量
    [G, W, P, a] = initializeVariables(n, d, L, candIdx, emptyMask);

    % 初始目标函数值
    F_prev = computeObjectiveTLSD2(X, Ymask, R, sumR, wVec, G, W, P, a, ...
        alpha, beta, gamma, lambda, tau, eps0, candIdx, noncandIdx, emptyMask);

    if isnan(F_prev) || isinf(F_prev)
        error('初始目标函数值异常: %.6e', F_prev);
    end

    obj = zeros(opts.maxIter, 1);

    if opts.verbose
        fprintf('初始目标函数值: %.6e\n', F_prev);
        if sum(emptyMask) > 0
            fprintf('检测到 %d 个空候选集样本，已自动处理\n', sum(emptyMask));
        end
        fprintf('\n');
    end

    % 早停策略
    best_obj = inf;
    patience_counter = 0;

    % ==================== 主循环 ====================
    for it = 1:opts.maxIter

        % ========== 预计算 ==========
        A = X .* G;
        S = A * W;

        % ========== P更新: CCCP + 投影 ==========
        Q = S + (gamma * L) * bsxfun(@times, wVec, P);

        P_old = P;
        P_new = projectP_rowsimplex_mask_optimized(Q, Ymask, candIdx, emptyMask);

        F_try = computeObjectiveTLSD2(X, Ymask, R, sumR, wVec, G, W, P_new, a, ...
            alpha, beta, gamma, lambda, tau, eps0, candIdx, noncandIdx, emptyMask);

        if F_try > F_prev + 1e-12
            [P_new, F_try] = dampingLineSearch(F_prev, P_old, P_new, ...
                X, Ymask, R, sumR, wVec, G, W, a, alpha, beta, gamma, lambda, ...
                tau, eps0, candIdx, noncandIdx, emptyMask, opts);
        end

        P = P_new;
        F_prev = F_try;

        % ========== W更新：梯度下降 + 投影 ==========
        [W, etaW_used] = updateW_projected(...
            X, G, W, P, candIdx, noncandIdx, ...
            alpha, tau, a, beta, emptyMask, n, d, L, eps0);

        % ========== G更新：梯度下降 + 投影 ==========
        [G, etaG_used] = updateG_projected(...
            X, G, W, P, candIdx, noncandIdx, ...
            beta, alpha, tau, a, emptyMask, n, d, L, eps0);

        % ========== a更新：梯度下降 + 投影 ==========
        [a, etaa_used] = updateA_projected(...
            G, W, R, sumR, a, beta, lambda, ...
            n, d, L, eps0, opts.a_min);

        % ========== 重新计算目标函数 ==========
        F_new = computeObjectiveTLSD2(X, Ymask, R, sumR, wVec, G, W, P, a, ...
            alpha, beta, gamma, lambda, tau, eps0, candIdx, noncandIdx, emptyMask);

        if isnan(F_new) || isinf(F_new)
            warning('目标函数异常，迭代%d: %.6e', it, F_new);
            obj(it) = F_prev;
        else
            F_prev = F_new;
            obj(it) = F_prev;
        end

        % ========== 早停策略 ==========
        if opts.useEarlyStopping
            if obj(it) < best_obj - opts.minImprove
                best_obj = obj(it);
                patience_counter = 0;
            else
                patience_counter = patience_counter + 1;
            end

            if patience_counter >= opts.patience
                if opts.verbose
                    fprintf('早停于迭代 %d（连续 %d 次无显著改进）\n', it, opts.patience);
                end
                obj = obj(1:it);
                return;
            end
        end

        % ========== 常规停止条件 ==========
        if it > 1
            rel = abs(obj(it) - obj(it-1)) / max(1, abs(obj(it-1)));
            if rel < opts.tol
                if opts.verbose
                    fprintf('收敛于迭代 %d，相对变化 %.2e\n', it, rel);
                end
                obj = obj(1:it);
                return;
            end
        end

        if opts.verbose && (mod(it, opts.printFreq) == 0 || it == 1)
            fprintf('迭代 %4d: obj = %.8e, ηW=%.2e, ηG=%.2e, ηa=%.2e\n', ...
                it, obj(it), etaW_used, etaG_used, etaa_used);
        end
    end

    obj = obj(1:opts.maxIter);
    if opts.verbose
        fprintf('达到最大迭代次数 %d\n', opts.maxIter);
    end
end

%% ========================================================================
%% 参数设置
%% ========================================================================
function opts = setDefaultOpts(opts)
    if ~isfield(opts, 'maxIter'),      opts.maxIter = 100; end
    if ~isfield(opts, 'tol'),          opts.tol = 1e-3; end
    if ~isfield(opts, 'tau'),          opts.tau = 10; end
    if ~isfield(opts, 'eps'),          opts.eps = 1e-12; end
    if ~isfield(opts, 'a_min'),        opts.a_min = 1e-6; end
    if ~isfield(opts, 'lsMax'),        opts.lsMax = 30; end
    if ~isfield(opts, 'lsShrink'),     opts.lsShrink = 0.5; end
    
    if ~isfield(opts, 'seed'),         opts.seed = 1; end
    if ~isfield(opts, 'verbose'),      opts.verbose = false; end
    if ~isfield(opts, 'printFreq'),    opts.printFreq = 10; end
    
    % 早停策略
    if ~isfield(opts, 'useEarlyStopping'), opts.useEarlyStopping = true; end
    if ~isfield(opts, 'patience'),     opts.patience = 15; end
    if ~isfield(opts, 'minImprove'),   opts.minImprove = 1e-5; end
end

%% ========================================================================
%% W更新：投影梯度下降（先梯度下降，再近端投影）
%% ========================================================================
function [W, eta_used] = updateW_projected(...
    X, G, W, P, candIdx, noncandIdx, ...
    alpha, tau, a, beta, emptyMask, n, d, L, eps0)

    % ===== 步骤1: 计算梯度 =====
    A = X .* G;
    S = A * W;
    [~, Dset] = setContrastLossGrad_LSE_optimized(S, candIdx, noncandIdx, emptyMask, tau);
    
    residual = S - P;
    gradW = (2/(n*L)) * (A' * residual) + (alpha/n) * (A' * Dset);
    
    % 数值检查
    if any(isnan(gradW(:))) || any(isinf(gradW(:)))
        warning('W梯度异常，跳过更新');
        eta_used = 0;
        return;
    end
    
    % ===== 步骤2: 计算Lipschitz常数 =====
    AtA_diag = sum(A.^2, 1)' + eps0;
    L_W = (2/(n*L)) * max(AtA_diag) + (alpha/n) * max(AtA_diag);
    
    % ===== 步骤3: 自适应学习率 =====
    eta = 1 / (L_W + eps0);
    eta_used = eta;
    
    % ===== 步骤4: 梯度下降 =====
    U = W - eta * gradW;
    
    % ===== 步骤5: 近端投影（组LASSO软阈值） =====
    W = proximal_group_lasso_vectorized(U, a, eta, beta, d, L, eps0);
end

%% ========================================================================
%% G更新：投影梯度下降（先梯度下降，再盒约束投影）
%% ========================================================================
function [G, eta_used] = updateG_projected(...
    X, G, W, P, candIdx, noncandIdx, ...
    beta, alpha, tau, a, emptyMask, n, d, L, eps0)

    % ===== 步骤1: 计算梯度 =====
    A = X .* G;
    S = A * W;
    [~, Dset] = setContrastLossGrad_LSE_optimized(S, candIdx, noncandIdx, emptyMask, tau);
    
    E = S - P;
    grad_fit = (2/(n*L)) * bsxfun(@times, X, E * W');
    grad_set = (alpha/n) * bsxfun(@times, X, Dset * W');
    
    s = sum(G, 1)';
    reg_col = (2*beta)/(d*n^2) * (s ./ (a + eps0));
    grad_reg = bsxfun(@times, ones(n, 1), reg_col');
    
    gradG = grad_fit + grad_set + grad_reg;
    
    % 数值检查
    if any(isnan(gradG(:))) || any(isinf(gradG(:)))
        warning('G梯度异常，跳过更新');
        eta_used = 0;
        return;
    end
    
    % ===== 步骤2: 计算Lipschitz常数 =====
    WWT_diag = sum(W.^2, 2) + eps0;
    X_sq_max = max(X(:).^2) + eps0;
    L_G = (2/(n*L)) * X_sq_max * max(WWT_diag) + ...
          (alpha/n) * X_sq_max * max(WWT_diag) + ...
          (2*beta)/(d*n^2) * max(1./(a + eps0));
    
    % ===== 步骤3: 自适应学习率 =====
    eta = 1 / (L_G + eps0);
    eta_used = eta;
    
    % ===== 步骤4: 梯度下降 =====
    U = G - eta * gradG;
    
    % ===== 步骤5: 投影到 [0,1] 盒约束 =====
    G = max(0, min(1, U));
end

%% ========================================================================
%% a更新：投影梯度下降（先梯度下降，再盒约束投影）
%% ========================================================================
function [a, eta_used] = updateA_projected(...
    G, W, R, sumR, a, beta, lambda, ...
    n, d, L, eps0, a_min)

    % ===== 步骤1: 计算梯度 =====
    s = sum(G, 1)';
    m = s / n;
    Wrow = sqrt(sum(W.^2, 2) + eps0);
    Ra = R * a;
    
    a_eps = a + eps0;
    c = (beta/d) * (m.^2 + Wrow / sqrt(L));
    grada = -(c ./ (a_eps.^2)) + (2*lambda/sumR) * Ra;
    
    % 数值检查
    if any(isnan(grada)) || any(isinf(grada))
        warning('a梯度异常，跳过更新');
        eta_used = 0;
        return;
    end
    
    % ===== 步骤2: 计算Lipschitz常数 =====
    c_max = max(c) + eps0;
    a_min_safe = max(min(a), eps0);
    
    R_norm = norm(R, 2);
    if isnan(R_norm) || isinf(R_norm)
        R_norm = 1.0;
    end
    
    L_a = 2 * c_max / (a_min_safe^3) + (2*lambda/sumR) * R_norm;
    
    % ===== 步骤3: 自适应学习率 =====
    eta = 1 / (L_a + eps0);
    eta_used = eta;
    
    % ===== 步骤4: 梯度下降 =====
    u = a - eta * grada;
    
    % ===== 步骤5: 投影到 [a_min, 1] 盒约束 =====
    a = max(a_min, min(1, u));
end

%% ========================================================================
%% 近端算子：组LASSO软阈值
%% ========================================================================
function W = proximal_group_lasso_vectorized(U, a, eta, beta, d, L, eps0)
    % 计算每行的L2范数
    nu = sqrt(sum(U.^2, 2)) + eps0;
    
    % 计算阈值
    th = eta * beta ./ (d * sqrt(L) * (a + eps0));
    
    % 软阈值缩放因子
    scale = max(1 - th ./ nu, 0);
    
    % 应用缩放
    W = bsxfun(@times, U, scale);
end

%% ========================================================================
%% 辅助函数
%% ========================================================================
function wVec = computeWeights_vectorized(candIdx, n, emptyMask)
    wVec = zeros(n, 1);
    ci = cellfun(@numel, candIdx);
    mask = (ci > 1) & (~emptyMask);
    wVec(mask) = ci(mask) ./ (ci(mask) - 1);
end

function [candIdx, noncandIdx, candMat, emptyMask] = getCandidateIndices(Ymask, n, L)
    candIdx = cell(n, 1);
    noncandIdx = cell(n, 1);
    candMat = false(n, L);
    emptyMask = false(n, 1);
    
    for i = 1:n
        c = find(Ymask(i, :));
        if isempty(c)
            emptyMask(i) = true;
            candIdx{i} = 1;
            noncandIdx{i} = [];
        else
            candIdx{i} = c;
            noncandIdx{i} = find(~Ymask(i, :));
            candMat(i, c) = true;
        end
    end
end

function [G, W, P, a] = initializeVariables(n, d, L, candIdx, emptyMask)
    G = ones(n, d);
    a = ones(d, 1);
    W = 0.01 * randn(d, L);

    P = zeros(n, L);
    for i = 1:n
        if ~emptyMask(i)
            c = candIdx{i};
            P(i, c) = 1 / numel(c);
        end
    end
end

function F = computeObjectiveTLSD2(X, Ymask, R, sumR, wVec, G, W, P, a, ...
    alpha, beta, gamma, lambda, tau, eps0, candIdx, noncandIdx, emptyMask)

    [n, d] = size(X);
    L = size(P,2);

    A = X .* G;
    S = A * W;

    diff = S - P;
    term1 = sum(diff(:).^2) / (n*L);

    setLoss = setContrastLoss_LSE_optimized(S, candIdx, noncandIdx, emptyMask, tau);
    term2 = alpha * setLoss / n;

    s = sum(G, 1)';
    m = s / n;
    term3 = beta * sum(m.^2 ./ (a + eps0)) / d;

    Wrow = sqrt(sum(W.^2, 2) + eps0);
    term4 = beta * sum(Wrow ./ (sqrt(L) * (a + eps0))) / d;

    P_rowNorm2 = sum(P.^2, 2);
    term5 = gamma * sum(wVec .* (1 - P_rowNorm2)) / n;

    term6 = lambda * (a' * (R * a)) / sumR;

    F = term1 + term2 + term3 + term4 + term5 + term6;
end

function R = computeR_cos2_psd(X, eps0)
    Xc = X - mean(X, 1);
    colNorm = sqrt(sum(Xc.^2, 1)) + eps0;
    Z = bsxfun(@rdivide, Xc, colNorm);
    C = Z' * Z;
    R = C .* C;
    R = 0.5 * (R + R');
end

function [loss, D] = setContrastLossGrad_LSE_optimized(S, candIdx, noncandIdx, emptyMask, tau)
    [n, L] = size(S);
    D = zeros(n, L);
    loss = 0;

    has_nc = ~cellfun(@isempty, noncandIdx);
    
    for i = 1:n
        if emptyMask(i)
            continue;
        end
        
        C  = candIdx{i};
        sc = S(i, C);
        [u, pic] = lse_and_softmax(sc, tau);

        if ~has_nc(i)
            continue;
        end

        NC = noncandIdx{i};
        sn = S(i, NC);
        [v, pin] = lse_and_softmax(sn, tau);

        x = v - u;
        loss = loss + softplus(x);
        sig = 1 / (1 + exp(-x));

        D(i, NC) = sig * pin;
        D(i, C)  = -sig * pic;
    end
end

function loss = setContrastLoss_LSE_optimized(S, candIdx, noncandIdx, emptyMask, tau)
    n = size(S, 1);
    loss = 0;
    
    has_nc = ~cellfun(@isempty, noncandIdx);

    for i = 1:n
        if emptyMask(i) || ~has_nc(i)
            continue;
        end
        
        u = lse_only(S(i, candIdx{i}), tau);
        v = lse_only(S(i, noncandIdx{i}), tau);
        loss = loss + softplus(v - u);
    end
end

function [lse, pi] = lse_and_softmax(s, tau)
    x = tau * s(:);
    mx = max(x);
    ex = exp(x - mx);
    Z = sum(ex);
    lse = (mx + log(Z)) / tau;
    pi = (ex / Z)';
end

function lse = lse_only(s, tau)
    x = tau * s(:);
    mx = max(x);
    lse = (mx + log(sum(exp(x - mx)))) / tau;
end

function y = softplus(x)
    y = max(x, 0) + log1p(exp(-abs(x)));
end

function [P_new, F_try] = dampingLineSearch(F_prev, P_old, P_new, ...
    X, Ymask, R, sumR, wVec, G, W, a, alpha, beta, gamma, lambda, ...
    tau, eps0, candIdx, noncandIdx, emptyMask, opts)

    t = 1.0;
    for ls = 1:opts.lsMax
        P_mix = (1 - t) * P_old + t * P_new;
        F_mix = computeObjectiveTLSD2(X, Ymask, R, sumR, wVec, G, W, P_mix, a, ...
            alpha, beta, gamma, lambda, tau, eps0, candIdx, noncandIdx, emptyMask);

        if F_mix <= F_prev + 1e-12
            P_new = P_mix;
            F_try = F_mix;
            return;
        end
        t = t * opts.lsShrink;
    end
    F_try = F_prev;
end

function P = projectP_rowsimplex_mask_optimized(Q, Ymask, candIdx, emptyMask)
    [n, L] = size(Q);
    P = zeros(n, L);

    for i = 1:n
        if emptyMask(i)
            continue;
        end
        
        C = candIdx{i};
        qi = Q(i, C);
        pi = proj_simplex_fast(qi);
        P(i, C) = pi;
    end

    P = P .* full(Ymask);
end

function p = proj_simplex_fast(q)
    q = q(:);
    m = length(q);

    [qs, ~] = sort(q, 'descend');
    cssv = cumsum(qs) - 1;
    ind = (1:m)';
    rho = find(qs > cssv ./ ind, 1, 'last');

    if isempty(rho)
        theta = 0;
    else
        theta = cssv(rho) / rho;
    end

    p = max(q - theta, 0);
    s = sum(p);
    if s <= 1e-14
        p = ones(m, 1) / m;
    else
        p = p / s;
    end

    p = p';
end