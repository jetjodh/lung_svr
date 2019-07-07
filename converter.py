from nilearn.input_data import NiftiMasker
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from nilearn import image
from nilearn.plotting import plot_img, show
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
import glob

data = pd.read_csv('TrainingSet_metaData.csv')
#print(data.head(5))

img = image.load_img("TrainingSet_Masks\CTR_TRN_001.nii")
plot_img(img,display_mode="z", cut_coords=[-9],
         vmin=.42, cmap='hot', threshold=.2, black_bg=False)
plot_img("TrainingSet_1_of_2\CTR_TRN_001.nii")
show()
print(img.shape)

#masked_data = apply_mask('TrainingSet_Masks\CTR_TRN_001.nii', img)
#masker = NiftiMasker(mask_img='TrainingSet_Masks\CTR_TRN_001.nii', standardize=True)
#fmri_masked = masker.fit_transform('TrainingSet_Masks\CTR_TRN_001.nii')

y = data.SVR_Severity
severity = {'HIGH': 1,'LOW': 0} 
y = [severity[item] for item in y] 
#print(y)

x = data.drop(['SVR_Severity','Filename'],axis=1)

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(x, y)
importances = clf.feature_importances_

feature_importances = pd.DataFrame(importances,
                                   index = x.columns,
                                   columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances)

X = []
for filename in glob.glob('G:\ImageCliff\TrainingSet_1_of_2\*.nii', recursive=True):
    X.append(filename)

for filename in glob.glob('G:\ImageCliff\TrainingSet_2_of_2\*.nii', recursive=True):
    X.append(filename)

for i in X:
    i = i.get_data()

print(X[7])

