from data_utils import load_data
from hybrid_model import build_bi_cnn_banet_model,build_bi_cnn_banet_angle_model
from model_utils import model_pipeline,plot_history
import numpy as np

train_participant_num = ["C56D","C93D","C382D","C382N","C544D","C709N","C788N","P113D","P113N","P191D","P191N","P299D","P299N","P300D","P336D","P492D","P492N","P531N","P699D","P699N","P890N","P921D","P921N"]
valid_participant_num = ["C67D","C202D","C202N","C256D","C256N","P54D","P54N","P342D","P342N","P487D","P487N","P649N"]

X_train, y_train = load_data(train_participant_num, 'train', downsampling=True,angle_energy=False,augment=False)
X_valid, y_valid = load_data(valid_participant_num, 'validation')

print(X_train.shape,np.unique(y_train[:,0],return_counts=True))

print(X_valid.shape,np.unique(y_valid[:,0],return_counts=True))

model = build_bi_cnn_banet_model()
y_pred, y_true, H = model_pipeline(model, X_train, y_train, X_valid, y_valid,epoch=30)
plot_history(H)

