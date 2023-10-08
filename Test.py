# GENERAL
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# PATH PROCESS
import os
import os.path
from pathlib import Path
import glob
# IMAGE PROCESS
from PIL import Image
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.applications.vgg16 import preprocess_input, decode_predictions
# SCALER & TRANSFORMATION
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import regularizers
from sklearn.preprocessing import LabelEncoder
# ACCURACY CONTROL
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
# OPTIMIZER
from keras.optimizers import RMSprop, Adam, Optimizer, Optimizer
# MODEL LAYERS
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, MaxPooling2D, BatchNormalization, \
                        Permute, TimeDistributed, Bidirectional, GRU, SimpleRNN, LSTM, GlobalAveragePooling2D, SeparableConv2D
from keras import models
from keras import layers
import tensorflow as tf
from keras.applications import VGG16, VGG19, inception_v3
from keras import backend as K
from keras.utils import plot_model
# SKLEARN CLASSIFIER
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
# IGNORING WARNINGS
from warnings import filterwarnings
filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning) 
filterwarnings("ignore", category=UserWarning)

# Path and Labels
Fire_Dataset_Path = Path("Dataset\Training and Validation")  # ดึงDataset มาใช้

JPG_Path = list(Fire_Dataset_Path.glob(r"*/*.jpg"))  # กำหนดDirของไฟล์jpgทุกรูป

JPG_Labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], JPG_Path))  # ตั้งค่าLabelให้โดยดูLabelจากชื่อไฟล์แม่

print("FIRE: ", JPG_Labels.count("fire"))
print("NO_FIRE: ", JPG_Labels.count("nofire"))

# Transfrom to series
JPG_Path_Series = pd.Series(JPG_Path, name="JPG").astype(str)
JPG_Labels_Series = pd.Series(JPG_Labels, name="CATEGORY")

Main_Train_Data = pd.concat([JPG_Path_Series, JPG_Labels_Series], axis=1)
# print(Main_Train_Data.head(-1))

# Shuffle Train data
Main_Train_Data = Main_Train_Data.sample(frac=1).reset_index(drop=True)

Test_Generator = ImageDataGenerator(rescale=1./255)
Train_Data, Test_Data = train_test_split(Main_Train_Data, train_size=0.9, random_state=42, shuffle=True)

Test_IMG_Set = Test_Generator.flow_from_dataframe(dataframe=Test_Data,
                                                  x_col="JPG",
                                                  y_col="CATEGORY",
                                                  color_mode="rgb",
                                                  class_mode="categorical",
                                                  batch_size=32)

loaded_model = tf.keras.models.load_model('rcnn_model.h5')
print("Loaded model summary is ")
print(loaded_model.summary())

Model_Results = loaded_model.evaluate(Test_IMG_Set)
print("LOSS:  " + "%.4f" % Model_Results[0])
print("ACCURACY:  " + "%.2f" % Model_Results[1])

# Test
Prediction_One = loaded_model.predict(Test_IMG_Set)
Prediction_Class_One = np.argmax(Prediction_One, axis=1)
accuracy = accuracy_score(Test_IMG_Set.classes, Prediction_Class_One)
print("Accuracy : {:.2f}%".format(accuracy * 100))

fig, axes = plt.subplots(nrows=8,
                         ncols=8,
                         figsize=(20, 20),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(cv2.imread(Test_Data["JPG"].iloc[i]))
    ax.set_title(f"TEST:{Test_Data.CATEGORY.iloc[i]}\n PREDICTION:{Prediction_Class_One[i]}")
plt.tight_layout()
plt.show()
