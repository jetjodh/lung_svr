import nibabel
import glob
X = []
img_lst = []

for filename in glob.glob('G:\ImageCliff\TrainingSet_2_of_2\*.nii', recursive=False):
    img_lst.append(filename)
i=0
for s in range(119):
    img = nibabel.load(img_lst[s])
    X.append(img.get_fdata())
    i=+s
    print(i)
print(X[7])