function B = columnMinMaxNormalization(A)
    % 输入：
    %   A - 待归一化的矩阵，维度为 m x n。

    % 计算每列的最小值和最大值
    minValues = min(A,[],1);
    maxValues = max(A,[],1);
    epsilon = 1e-12; % 设置一个小的正数作为阈值
    diffValues = max(maxValues - minValues + epsilon, epsilon);

    % 避免分母为零的情况（如果最大值等于最小值） % 设置一个小的正数作为阈值

    % 进行归一化计算
    B = (A - minValues) ./ diffValues;

    % 返回归一化后的矩阵B
end