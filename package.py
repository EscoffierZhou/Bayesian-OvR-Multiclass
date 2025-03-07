import numpy as np
import pandas as pd
import torch
import tqdm
import time
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import torch.utils.data as data_utils
from datetime import datetime
from sklearn.impute import KNNImputer
from torch.distributions import Normal
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from package import KNNImputer,tqdm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from bayesian_formula import getProbabilities, action  # 仅导入需要的函数
from Data_processing import (
    feature_lists,impute_numerical,impute_categorical,encoder
)
from train import *
from bayesian_formula import *
from after_exploring import *
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']