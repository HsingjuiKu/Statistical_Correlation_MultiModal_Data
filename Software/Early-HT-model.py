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


#X,Y,Z Coordinates
X_train_pose = X_train[:, :, 0:66]
X_valid_pose = X_valid[:, :, 0:66]
#sEMG
X_train_sEMG = X_train[:, :, 66:70]
X_valid_sEMG = X_valid[:, :, 66:70]


input_shape_pose = (X_train_pose.shape[1], X_train_pose.shape[2])
input_shape_sEMG = (X_train_sEMG.shape[1], X_train_sEMG.shape[2])
num_classes = y_train.shape[1]

model_pose = stacked_lstm(input_shape_pose, num_classes)
model_sEMG = stacked_lstm(input_shape_sEMG, num_classes)

y_pred_pose, y_true_pose, H_pose = model_pipeline(model_pose, X_train_pose, y_train, X_valid_pose, y_valid)
y_pred_sEMG, y_true_sEMG, H_sEMG = model_pipeline(model_sEMG, X_train_sEMG, y_train, X_valid_sEMG, y_valid)

pose_correlations = []
for i in range(X_train_pose.shape[2]):  # 遍历每个特征维度
    repeated_y_train = np.repeat(y_train, X_train_pose.shape[1])
    coef, p_value = spearmanr(X_train_pose[:, :, i].flatten(), repeated_y_train[:X_train_pose[:, :, i].flatten().shape[0]])
    pose_correlations.append((coef, p_value))

sEMG_correlations = []
for i in range(X_train_sEMG.shape[2]):  # 遍历每个特征维度
    repeated_y_train = np.repeat(y_train, X_train_sEMG.shape[1])
    coef, p_value = spearmanr(X_train_sEMG[:, :, i].flatten(), repeated_y_train[:X_train_sEMG[:, :, i].flatten().shape[0]])
    sEMG_correlations.append((coef, p_value))



weight_pose_abs_avg = np.mean(np.abs([coef for coef, p in pose_correlations]))
weight_sEMG_abs_avg = np.mean(np.abs([coef for coef, p in sEMG_correlations]))


total_weight_avg = weight_pose_abs_avg + weight_sEMG_abs_avg
normalized_weight_pose = weight_pose_abs_avg / total_weight_avg
normalized_weight_sEMG = weight_sEMG_abs_avg / total_weight_avg

final_pred = y_pred_pose * normalized_weight_pose + y_pred_sEMG * normalized_weight_sEMG

final_pred_class = np.round(final_pred).astype(int)

print(classification_report(y_true_pose, final_pred_class))
print(confusion_matrix(y_true_pose, final_pred_class))