from data_utils import load_data, flatten_data
from early_model import stacked_lstm
from model_utils import model_pipeline, plot_history
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import ttest_ind
from scipy.stats import spearmanr

train_participant_num = ["C56D", "C93D", "C382D", "C382N", "C544D", "C709N", "C788N", "P113D", "P113N", "P191D", "P191N", "P299D", "P299N", "P300D", "P336D", "P492D", "P492N", "P531N", "P699D", "P699N", "P890N", "P921D", "P921N"]
valid_participant_num = ["C67D", "C202D", "C202N", "C256D", "C256N", "P54D", "P54N", "P342D", "P342N", "P487D", "P487N", "P649N"]

X_train, y_train = load_data(train_participant_num, 'train', downsampling=True, angle_energy=False, augment=False)
X_valid, y_valid = load_data(valid_participant_num, 'validation')

num_classes = y_train.shape[1]

X_train_XYZ = X_train[:, :, :66]  # XYZ坐标
X_train_sEMG = X_train[:, :, 66:70]  # sEMG数据

correlations_XYZ = []
correlations_sEMG = []

n_samples = y_train.shape[0]

for i in range(22):
    # 获取每个XYZ坐标的展平后的数组
    X_flat = X_train[:, :, i].flatten()
    Y_flat = X_train[:, :, i+22].flatten()
    Z_flat = X_train[:, :, i+44].flatten()

    # 为了匹配X_flat, Y_flat, Z_flat的长度，我们需要正确地重复y_train
    # 重复y_train以匹配每个样本的时间步骤
    y_repeated = np.repeat(y_train, X_train.shape[1])

    # 计算相关性
    coef_X, _ = spearmanr(X_flat, y_repeated[:len(X_flat)])
    coef_Y, _ = spearmanr(Y_flat, y_repeated[:len(Y_flat)])
    coef_Z, _ = spearmanr(Z_flat, y_repeated[:len(Z_flat)])

    # 计算平均相关系数
    avg_coef = np.mean([coef_X, coef_Y, coef_Z])
    correlations_XYZ.append(avg_coef)

# 对sEMG数据计算相关性
for i in range(4):
    sEMG_flat = X_train_sEMG[:, :, i].flatten()
    y_repeated_sEMG = np.repeat(y_train, X_train_sEMG.shape[1])
    coef_sEMG, _ = spearmanr(sEMG_flat, y_repeated_sEMG[:len(sEMG_flat)])
    correlations_sEMG.append(coef_sEMG)

# print("Correlations for XYZ coordinates:", correlations_XYZ)
# print("Correlations for sEMG signals:", correlations_sEMG)


y_preds = []
for i in range(22):
    X_train_xyz = X_train[:, :, [i, i+22, i+44]]
    X_valid_xyz = X_valid[:, :, [i, i+22, i+44]]
    model_xyz = stacked_lstm((X_train_xyz.shape[1], 3), num_classes)
    _, y_pred_xyz, H_xyz = model_pipeline(model_xyz, X_train_xyz, y_train, X_valid_xyz, y_valid)
    y_preds.append(y_pred_xyz)


X_train_sEMG = X_train[:, :, 66:70]
X_valid_sEMG = X_valid[:, :, 66:70]
model_sEMG = stacked_lstm((X_train_sEMG.shape[1], X_train_sEMG.shape[2]), num_classes)
_, y_pred_sEMG, H_sEMG = model_pipeline(model_sEMG, X_train_sEMG, y_train, X_valid_sEMG, y_valid)
y_preds.append(y_pred_sEMG)

mean_correlation_sEMG = np.mean(correlations_sEMG)
correlations_XYZ.append(mean_correlation_sEMG)
correlations = correlations_XYZ



# 计算绝对值并设置最大阈值
abs_correlations = np.abs(correlations)
max_threshold = 0.5

# 如果绝对值大于最大阈值，则将其设置为最大阈值
clipped_weights = np.clip(abs_correlations, None, max_threshold)

# 归一化处理，确保所有权重之和为1
normalized_weights = clipped_weights / np.sum(clipped_weights)

# 为了保证其他权重平均分配，对于未被强制设置为最大阈值的权重进行再次归一化
# 计算未被强制设置为阈值的权重之和
remaining_weight_sum = np.sum(normalized_weights[normalized_weights < max_threshold])

# 对未被设置为最大阈值的权重进行再次归一化
normalized_weights[normalized_weights < max_threshold] /= remaining_weight_sum

# 确保归一化后的权重之和为1
normalized_weights /= np.sum(normalized_weights)


# 加权平均预测
weighted_preds = np.zeros(y_preds[0].shape)  # 初始化加权预测数组

for i, y_pred in enumerate(y_preds):
    weighted_preds += y_pred * normalized_weights[i]  # 加权预测

final_preds = np.round(weighted_preds).astype(int)

print(classification_report(y_valid, final_preds))
print(confusion_matrix(y_valid, final_preds))