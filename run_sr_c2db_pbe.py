#!/usr/bin/env python3
"""
运行C2DB PBE带隙的符号回归
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pysr import PySRRegressor
import joblib

def split_data_by_target(csv_file, output_train_file, output_test_file, test_size=0.1, bins=5, normalize=True):
    """
    分割数据并可选择性地进行归一化
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # 如果有索引列，去掉
    if 'material_id' in df.columns:
        df = df.drop(columns=['material_id'])
    if 'formula' in df.columns:
        df = df.drop(columns=['formula'])
    
    print(f"Data shape: {df.shape}")
    print(f"Target column: 'target'")
    
    # 分层抽样
    df['bins'] = pd.cut(df['target'], bins=bins, labels=False)
    train_set, test_set = train_test_split(df, test_size=test_size, stratify=df['bins'], random_state=42)
    train_set = train_set.drop(columns=['bins'])
    test_set = test_set.drop(columns=['bins'])
    
    if normalize:
        print("Applying normalization...")
        
        # 分离特征和目标
        X_train = train_set.drop(columns=['target'])
        y_train = train_set['target']
        X_test = test_set.drop(columns=['target'])
        y_test = test_set['target']
        
        # 特征归一化 (MinMaxScaler to [0,1])
        feature_scaler = MinMaxScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
        
        # 目标归一化 (MinMaxScaler to [0,1])
        target_scaler = MinMaxScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
        
        # 重新组合数据
        train_set_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        train_set_scaled['target'] = y_train_scaled
        
        test_set_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        test_set_scaled['target'] = y_test_scaled
        
        # 保存归一化后的数据
        train_set_scaled.to_csv(output_train_file, index=False)
        test_set_scaled.to_csv(output_test_file, index=False)
        
        # 保存scaler以便后续反归一化
        scaler_file = csv_file.replace('.csv', '_scalers.pkl')
        joblib.dump({
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
            'feature_columns': list(X_train.columns)
        }, scaler_file)
        
        print(f"Normalization completed:")
        print(f"  Features: MinMaxScaler (range=[0,1])")
        print(f"  Target: MinMaxScaler (range=[0,1])")
        print(f"  Original target range: [{y_train.min():.4f}, {y_train.max():.4f}]")
        print(f"  Normalized target range: [{y_train_scaled.min():.4f}, {y_train_scaled.max():.4f}]")
        print(f"  Scalers saved to: {scaler_file}")
        
        return train_set_scaled, test_set_scaled
    else:
        # 不归一化，直接保存
        train_set.to_csv(output_train_file, index=False)
        test_set.to_csv(output_test_file, index=False)
        return train_set, test_set

def get_matrix(model, index, y):
    y_pred = model.predict(X, index=index)
    y_true = y
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, r2

def inverse_transform_predictions(predictions, scaler_file):
    """
    使用保存的scaler对预测结果进行反归一化
    """
    try:
        scalers = joblib.load(scaler_file)
        target_scaler = scalers['target_scaler']
        # 反归一化预测值
        predictions_original = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        return predictions_original
    except:
        print("Warning: Could not load scalers for inverse transformation")
        return predictions

def get_matrix_with_denormalization(model, index, y_test_normalized, y_test_original, scaler_file):
    """
    获取评估指标，包含反归一化的结果
    """
    # 在归一化数据上进行预测
    y_pred_normalized = model.predict(X, index=index)
    
    # 反归一化预测值
    y_pred_original = inverse_transform_predictions(y_pred_normalized, scaler_file)
    
    # 计算归一化数据的指标
    mae_normalized = mean_absolute_error(y_test_normalized, y_pred_normalized)
    r2_normalized = r2_score(y_test_normalized, y_pred_normalized)
    
    # 计算原始数据的指标
    mae_original = mean_absolute_error(y_test_original, y_pred_original)
    r2_original = r2_score(y_test_original, y_pred_original)
    
    return {
        'mae_normalized': mae_normalized,
        'r2_normalized': r2_normalized,
        'mae_original': mae_original,
        'r2_original': r2_original
    }

# MAE损失函数
objective2 = """
function mae_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    (prediction, completion) = eval_tree_array(tree, dataset.X, options)
    if !completion
        return L(Inf)
    end
    true_labels = dataset.y
    
    # 计算MAE损失
    mae_loss = sum(abs.(true_labels .- prediction)) / length(true_labels)
    
    return L(mae_loss)
end
"""

print("Starting Symbolic Regression for C2DB PBE Bandgap...")
print("Using normalized features and targets for better convergence")

test_result_dict = defaultdict(dict)
for random_seed in range(100):
    print(f"\n--- Running SR with random seed {random_seed} ---")
    
    # 使用清理后的C2DB PBE特征文件
    csv_file = 'c2db_pbe_features_minimal.csv'
    output_train_file = f'c2db_pbe_bandgap_train_{random_seed}.csv'
    output_test_file = f'c2db_pbe_bandgap_test_{random_seed}.csv'
    
    # 分割数据并归一化
    train_set, test_set = split_data_by_target(csv_file, output_train_file, output_test_file, normalize=True)
    
    # 准备训练数据 (归一化后的)
    X_train = np.array(train_set.iloc[:, :-1])  # 所有列除了target
    y_train = np.array(train_set.iloc[:, -1])   # target列
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Training target range: [{y_train.min():.4f}, {y_train.max():.4f}]")
    
    # 配置PySR
    model = PySRRegressor(
        niterations=1000,
        random_state=random_seed,
        binary_operators=["+", "-", "*", "/"],
        population_size=50,
        unary_operators=[
            "square",
            "cube", 
            "sqrt",
            "exp",
            "log",
            "inv(x) = 1/x",
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        loss_function=objective2,
        complexity_of_operators={
            "+": 1, "-": 1, "*": 2, "/": 2,
            "square": 2, "cube": 3, "sqrt": 2,
            "exp": 4, "log": 4, "inv": 2
        },
        maxsize=20,
        parsimony=0.0032,
    )
    
    print("Fitting PySR model...")
    model.fit(X_train, y_train)
    
    # 准备测试数据
    X_test = np.array(test_set.iloc[:, :-1])
    y_test_normalized = np.array(test_set.iloc[:, -1])
    
    # 加载原始测试目标值（用于评估真实性能）
    scaler_file = csv_file.replace('.csv', '_scalers.pkl')
    try:
        scalers = joblib.load(scaler_file)
        target_scaler = scalers['target_scaler']
        y_test_original = target_scaler.inverse_transform(y_test_normalized.reshape(-1, 1)).flatten()
    except:
        # 如果无法加载scaler，使用归一化的值
        y_test_original = y_test_normalized
    
    # 设置全局X用于get_matrix函数
    global X
    X = X_test
    
    print(f"Evaluating {len(model.equations)} equations...")
    
    # 评估每个方程
    for index in range(len(model.equations)):
        equation = model.equations['sympy_format'][index]
        metrics = get_matrix_with_denormalization(model, index, y_test_normalized, y_test_original, scaler_file)
        test_result_dict[random_seed][equation] = metrics
        
        if index < 3:  # 显示前3个最佳方程的性能
            print(f"  Equation {index}: {equation}")
            print(f"    R² (original): {metrics['r2_original']:.4f}")
            print(f"    MAE (original): {metrics['mae_original']:.4f}")
    
    # 保存结果
    np.save('test_result_dict_c2db_pbe.npy', test_result_dict)
    
    print(f"Seed {random_seed} completed")

print("\nSymbolic Regression for C2DB PBE completed!")
print("Results saved to: test_result_dict_c2db_pbe.npy")
