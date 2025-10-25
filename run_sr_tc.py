import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from pysr import PySRRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# def split_data_by_target(csv_file, output_train_file, output_test_file, test_size=0.1, random_state=42):
#     df = pd.read_csv(csv_file)
#     target_1 = df[df['target'] == 1]
#     target_0 = df[df['target'] == 0]
#     train_target_1, test_target_1 = train_test_split(target_1, test_size=test_size, random_state=random_state)
#     train_target_0, test_target_0 = train_test_split(target_0, test_size=test_size, random_state=random_state)
#     train_set = pd.concat([train_target_1, train_target_0])
#     test_set = pd.concat([test_target_1, test_target_0])
#     train_set.to_csv(output_train_file, index=False)
#     test_set.to_csv(output_test_file, index=False)
#     return train_set, test_set

def split_data_by_target(csv_file, output_train_file, output_test_file, test_size=0.1, bins=20):
    df = pd.read_csv(csv_file, index_col=0)
    df['bins'] = pd.cut(df['target'], bins=bins, labels=False)
    train_set, test_set = train_test_split(df, test_size=test_size, stratify=df['bins'], random_state=42)
    train_set = train_set.drop(columns=['bins'])
    test_set = test_set.drop(columns=['bins'])
    train_set.to_csv(output_train_file, index=False)
    test_set.to_csv(output_test_file, index=False)
    return train_set, test_set


def get_matrix(model,index,y):
    y_pred = model.predict(X, index=index)
    y_true = y
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, r2

objective1 = """
function accuracy_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    (prediction, completion) = eval_tree_array(tree, dataset.X, options)
    if !completion
        return L(Inf)
    end
    predicted_labels = prediction .>= 0.5 
    true_labels = dataset.y
    correct_predictions = sum(predicted_labels .== true_labels)
    accuracy = correct_predictions / length(true_labels)
    loss = 1.0 - accuracy
    return L(loss)
end
"""

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

# test_result_dict = defaultdict(dict)
# for random_seed in range(100):
#     csv_file = 'feature_df_large_all.csv'
#     output_train_file = f'feature_df_large_all_train_{random_seed}.csv'
#     output_test_file = f'feature_df_large_all_test_{random_seed}.csv'
#     train_set, test_set = split_data_by_target(csv_file, output_train_file, output_test_file)
#     X = np.array(train_set.iloc[:, 1:-1])
#     y = np.array(train_set.iloc[:, -1])
#     model = PySRRegressor(
#         niterations=1000,  # < Increase me for better results
#         random_state=random_seed,
#         binary_operators=["+", "-","*","/"],
#         population_size=50,
#         unary_operators=[
#             "square",
#             "cube",
#             "inv(x) = 1/x",
#         ],
#         extra_sympy_mappings={"inv": lambda x: 1 / x},
#         loss_function=objective1,
#     )
#     model.fit(X, y)
#     X = np.array(test_set.iloc[:, 1:-1])
#     y = np.array(test_set.iloc[:, -1])
#     for index in range(len(model.equations)):
#         test_result_dict[random_seed][model.equations['sympy_format'][index]] = get_matrix(model, index, y)
#     np.save('test_result_dict1.npy',test_result_dict)

test_result_dict = defaultdict(dict)
for random_seed in range(100):
    csv_file = 'feature_df_large_tc.csv'
    output_train_file = f'feature_df_large_tc_train_{random_seed}.csv'
    output_test_file = f'feature_df_large_tc_test_{random_seed}.csv'
    train_set, test_set = split_data_by_target(csv_file, output_train_file, output_test_file)
    X = np.array(train_set.iloc[:, 1:-1])
    y = np.array(train_set.iloc[:, -1])
    model = PySRRegressor(
        niterations=1000,  # < Increase me for better results
        random_state=random_seed,
        binary_operators=["+", "-","*","/"],
        population_size=50,
        unary_operators=[
            "square",
            "cube",
            "inv(x) = 1/x",
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        loss_function=objective2,
    )
    model.fit(X, y)
    X = np.array(test_set.iloc[:, 1:-1])
    y = np.array(test_set.iloc[:, -1])
    for index in range(len(model.equations)):
        test_result_dict[random_seed][model.equations['sympy_format'][index]] = get_matrix(model, index, y)
    np.save('test_result_dict_tc.npy',test_result_dict)