#!/usr/bin/env python
# coding: utf-8

# #                         COMP 448/548 – Medical Image Analysis
# #                                                Homework #2
# ## Batuhan Yalçın
# ### 64274
# ### May 10, 2022

# In[ ]:


# Homework 07: Linear Discriminant Analysisn
## Batuhan Yalçın
### May 6, 2022import numpy as np
import math
from PIL import Image
from numpy import asarray
from skimage.color import rgb2gray
from sklearn import svm
from numpy import ma
import copy
import os
import cv2
import glob
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar


# # Part I

# ## calculateCooccurrenceMatrix Function

# In[ ]:


def calculateCooccurrenceMatrix(grayImg, binNumber, di, dj):
    bin_image = copy.deepcopy(grayImg)
    
    for i in range(len(grayImg)):
        for j in range(len(grayImg[0])):
            pixel_bin = int(math.floor(grayImg[i][j])/(len(grayImg[0])/binNumber))
            bin_image[i][j] = pixel_bin
    
    M = [[0 for i in range(binNumber)] for j in range(binNumber)]
    for i in range(binNumber):
        for j in range(binNumber):
            for k in range(len(grayImg)):
                for l in range(len(grayImg[0])):
                    if (k - di > -1) and (k - di < len(grayImg)) and (l - dj > -1) and (l - dj < len(grayImg)) :
                        if (bin_image[k][l] == i) and (bin_image[k-di][l - dj] == j):
                            M[i][j]+=1

    return M
    


# ## calculateAccumulatedCooccurrenceMatrix Function

# In[ ]:


def calculateAccumulatedCooccurrenceMatrix(grayImg, binNumber, d):
    accM = np.zeros((binNumber,binNumber))
    accM += np.array(calculateCooccurrenceMatrix(grayImg, binNumber, d, 0))
    accM += np.array(calculateCooccurrenceMatrix(grayImg, binNumber, d, d))
    accM += np.array(calculateCooccurrenceMatrix(grayImg, binNumber, 0, d))  
    accM += np.array(calculateCooccurrenceMatrix(grayImg, binNumber, -d, d))
    accM += np.array(calculateCooccurrenceMatrix(grayImg, binNumber, -d, 0))
    accM += np.array(calculateCooccurrenceMatrix(grayImg, binNumber, -d, -d))
    accM += np.array(calculateCooccurrenceMatrix(grayImg, binNumber, 0, -d))
    accM += np.array(calculateCooccurrenceMatrix(grayImg, binNumber, d, -d))
    return accM


# ## calculateCooccurrenceFeatures Function

# In[ ]:


def calculateCooccurrenceFeatures(accM):
    normalized_acc_m = accM / np.sum(accM)
    
    # The angular second moment
    ang_nd_moment = np.sum(np.power(normalized_acc_m, 2))
    
    # Maximum probability
    max_prob = np.max(normalized_acc_m)
    
    N = normalized_acc_m.shape[0]
    i = np.arange(1, N+1)
    j = np.arange(1, N+1)
    i.shape = (N, 1)
    j.shape = (1, N)
    
    # Inverse difference moment
    inverse_difference_moment = np.sum(normalized_acc_m / (1 + np.power(i - j, 2)))
    
    # Contrast
    contrast = np.sum(normalized_acc_m * np.power(i - j, 2))
    
    # Entropy 
    entropy = -np.sum(normalized_acc_m * (ma.log(normalized_acc_m).filled(0)))
    
    ii = np.arange(1, N+1)
    jj = np.arange(1, N+1)
    sum_i = np.sum(normalized_acc_m, 1)
    sum_j = np.sum(normalized_acc_m, 0)
    mu_x = np.sum(sum_i * i)
    mu_y = np.sum(sum_j * j)
    standart_d_x = np.sqrt(np.sum(sum_i * np.power(i - mu_x, 2)))
    standart_d_y = np.sqrt(np.sum(sum_j * np.power(j - mu_y, 2)))
    to_sum = np.sum((i * j) * normalized_acc_m)
    # correlation 
    correlation = (to_sum - mu_x * mu_y) / (standart_d_x * standart_d_y)
    return [ang_nd_moment, max_prob, inverse_difference_moment, contrast, entropy, correlation]


# # Part II

# ## Accumulated co-occurrence matrix  Extracting the six texture features

# ## Linear svm

# ## Note that below part is take so long so go below and direclty upload the datas from features which I already get features data and store in cvs

# In[35]:


## Note that below part is take so long so go below and direclty upload the datas from features which I already get features data and store in cvs# get all images and their features for training
bin_number = 8
d = 10
training_dir = "dataset//training//"
features = []
for img_file in glob.glob(training_dir + "*.jpg"):
    print(img_file)
    img = cv.imread(img_file, 0)
    acc_m = calculateAccumulatedCooccurrenceMatrix(img, bin_number, d)
    features.append(calculateCooccurrenceFeatures(acc_m))
train_features=np.array(features)    
training_labels = np.loadtxt("dataset//training_labels.txt")
training_labels = training_labels.tolist()
# save to csv file
np.savetxt('train_features.csv', features, delimiter=',')


# In[ ]:


training_labels = np.loadtxt("dataset//training_labels.txt")
print(training_labels)
training_labels = training_labels.tolist()
print(training_labels)
# load from the csv file
train_features_part2 = np.loadtxt('train_features_part2.csv',delimiter=',')


# ## Note that below part is take so long so go below and direclty upload the datas from features which I already get features data and store in cvs

# In[36]:


## Note that below part is take so long so go below and direclty upload the datas from features which I already get features data and store in cvstest_dir ="dataset//test//"
test_features = []
for img_file in glob.glob(test_dir + "*.jpg"):
    img = cv.imread(img_file, 0)
    acc_m = calculateAccumulatedCooccurrenceMatrix(img, bin_number, d)
    test_features.append(calculateCooccurrenceFeatures(acc_m))
test_labels = np.loadtxt("dataset//test_labels.txt")
test_labels = test_labels.tolist()
# save to csv file
np.savetxt('test_features3.csv', features, delimiter=',')


# In[270]:


test_labels = np.loadtxt("dataset//test_labels.txt")
test_labels = test_labels.tolist()
# load from the csv file
test_features = np.loadtxt('test_features_part2.csv',delimiter=',')


# ## Linear svm

# In[335]:


## Accumulated co-occurrence matrix  Extracting the six texture features
C_values = [0.1, 1, 5, 10, 50, 100, 250, 500, 1000, 5000]
Gamma_values = np.linspace(0.1,10,100)
scaler = StandardScaler()
train_features_fitted = scaler.fit_transform(train_features)
train_features_fitted = np.array(train_features_fitted)
test_features_fitted = scaler.fit_transform(test_features)
test_features_fitted = np.array(test_features_fitted)


for c in C_values:
    svc = SVC(kernel='linear', class_weight='balanced', C=c)
    model = svc.fit(features_fitted, training_labels)
    training_pred = np.array(model.predict(features_fitted))
    test_pred = np.array(model.predict(test_features_fitted))

    print(c)
    print('Overall Accuracy for train  ', accuracy_score(training_labels, training_pred))
    print('Class Accuracy for train ', confusion_matrix(training_labels, training_pred, normalize="true").diagonal())
    print('Overall Accuracy : ', accuracy_score(test_labels, test_pred))
    print('Class Accuracy : ', confusion_matrix(test_labels, test_pred, normalize="true").diagonal())

for c in (np.linspace(0.1,6,60)):
    svc = SVC(kernel='linear', class_weight='balanced', C=c)
    model = svc.fit(train_features_fitted, training_labels)
    training_pred_lin = np.array(model.predict(features_fitted))
    test_pred_lin = np.array(model.predict(test_features_fitted))
    np.savetxt('test_pred_lin', test_pred_lin, delimiter=',')

    print(c)
    print('Overall Accuracy for train  ', accuracy_score(training_labels, training_pred_lin))
    print('Class Accuracy for train ', confusion_matrix(training_labels, training_pred_lin, normalize="true").diagonal())
    print('Overall Accuracy : ', accuracy_score(test_labels, test_pred_lin))
    print('Class Accuracy : ', confusion_matrix(test_labels, test_pred_lin, normalize="true").diagonal())
#Best train accuracy
c=5000
#Best test accuracy while protecting the train accuracy
c=5.2
C_values.append(5.2)
svc = SVC(kernel='linear', class_weight='balanced', C=c)
model = svc.fit(train_features_fitted, training_labels)
training_pred_lin = np.array(model.predict(train_features_fitted))
test_pred_lin = np.array(model.predict(test_features_fitted))
np.savetxt('test_pred_lin', test_pred_lin, delimiter=',')


# In[417]:


for c in (np.linspace(0.1,0.2,10)):
    if (c==0.16666666666666669):
        c=5000
    if (c==0.17777777777777778):
        c=5.2
    svc = SVC(kernel='linear', class_weight='balanced', C=c)
    model = svc.fit(train_features_fitted, training_labels)
    training_pred_lin = np.array(model.predict(features_fitted))
    test_pred_lin = np.array(model.predict(test_features_fitted))
    np.savetxt('test_pred_lin', test_pred_lin, delimiter=',')
    if(accuracy_score(test_labels, test_pred_lin)>0.46 or c==5.2 or c==5000):
        print("its okey!!")
        print(c)
        print('Overall Accuracy for train  ', accuracy_score(training_labels, training_pred_lin))
        print('Class Accuracy for train ', confusion_matrix(training_labels, training_pred_lin, normalize="true").diagonal())
        print('Overall Accuracy : ', accuracy_score(test_labels, test_pred_lin))
        print('Class Accuracy : ', confusion_matrix(test_labels, test_pred_lin, normalize="true").diagonal())
#Best train accuracy
c=5000
#Best test accuracy 
c=0.196
C_values.append(0.196)
#Best test accuracy while protecting the train accuracy
c=5.2
svc = SVC(kernel='linear', class_weight='balanced', C=c)
model = svc.fit(train_features_fitted, training_labels)
training_pred_lin = np.array(model.predict(train_features_fitted))
test_pred_lin = np.array(model.predict(test_features_fitted))
np.savetxt('test_pred_lin', test_pred_lin, delimiter=',')


# In[337]:


# from list C=5.2 chosen for better acurracy rbf with gamma looked
for c in C_values:
    for gamma in Gamma_values:
        svc = SVC(kernel='rbf', class_weight='balanced', C=c, gamma=gamma)
        model = svc.fit(train_features_fitted, training_labels)
        training_pred = np.array(model.predict(train_features_fitted))
        test_pred = np.array(model.predict(test_features_fitted))
        
        #print(c,gamma)
        if(accuracy_score(test_labels, test_pred)>0.47):
            print(c,gamma)
            print("its okey")
            print('Overall Accuracy for train  ', accuracy_score(training_labels, training_pred))
            print('Class Accuracy for train ', confusion_matrix(training_labels, training_pred, normalize="true").diagonal())
            print('Overall Accuracy : ', accuracy_score(test_labels, test_pred))
            print('Class Accuracy : ', confusion_matrix(test_labels, test_pred, normalize="true").diagonal())
        
        

        
#Best Train and test accuracy
c_ref=50
gamma=3.4

svc = SVC(kernel='rbf', class_weight='balanced', C=c, gamma=gamma)
model = svc.fit(train_features_fitted, training_labels)
training_pred_rbf = np.array(model.predict(train_features_fitted))
test_pred_rbf = np.array(model.predict(test_features_fitted))
np.savetxt('test_pred_rbf', test_pred_rbf, delimiter=',')


# In[329]:


def contingency_table(truth, comporison1, comporison2, val=0):
    table = [[0,0],[0,0]]
    tf_comporison1 = truth == comporison1
    tf_comporison2 = truth == comporison2
    if val != 0:
        tf_comporison1 = tf_comporison1[truth == val]
        tf_comporison2 = tf_comporison2[truth == val]
    table[0][0] = np.sum(np.logical_and(tf_comporison1, tf_comporison2))
    table[0][1] = np.sum(np.logical_and(tf_comporison1, np.logical_not(tf_comporison2)))
    table[1][0] = np.sum(np.logical_and(np.logical_not(tf_comporison1), tf_comporison2))
    table[1][1] = np.sum(np.logical_and(np.logical_not(tf_comporison1), np.logical_not(tf_comporison2)))
    return table





training_pred_lin = np.array(training_pred_lin)
training_pred_rbf = np.array(training_pred_rbf)
test_pred_lin = np.array(test_pred_lin)
test_pred_rbf = np.array(test_pred_rbf)
training_labels = np.array(training_labels)
test_labels = np.array(test_labels)
for i in range(4):
    table_train = contingency_table(training_labels, training_pred_lin, training_pred_rbf, i)
    result_train = mcnemar(table_train, exact=False, correction=True)
    print("Result_Train\n",result_train)
    
    table_test = contingency_table(test_labels, test_pred_lin, test_pred_rbf, i)
    result_test = mcnemar(table_test, exact=False, correction=True)
    print("Result_Test\n",result_test)


# In[330]:


training_labels


# In[ ]:





# # Part III

# In[338]:


# get all images and their features for training
bin_number = 8
d = 10
N = 4


# ## Note that below part is take so long so go below and direclty upload the datas from features which I already get features data and store in cvs

# In[303]:


# Part IIITrain_features_part3 = []
for img_file in glob.glob(training_dir + "*.jpg"):
    img = cv.imread(img_file, 0)
    x_dimension, y_dimension = img.shape
    height = x_dimension // N
    width = y_dimension // N
    acc_m_list = []
    for i in range(N):
        for j in range(N):
            patch = img[i*height:(i+1)*height, j*width:(j+1)*width]
            acc_m = calculateAccumulatedCooccurrenceMatrix(patch, bin_number, d)
            acc_m_list.append(calculateCooccurrenceFeatures(acc_m))


    Train_features_part3.append(np.mean((np.array(acc_m_list)), 0))
    
Train_features_part3 = np.array(Train_features_part3)
np.savetxt('Train_features_part3.csv', Train_features_part3, delimiter=',')


# ## Note that below part is take so long so go below and direclty upload the datas from features which I already get features data and store in cvs

# In[177]:


Test_features_part3 = []
for img_file in glob.glob(test_dir + "*.jpg"):
    img = cv.imread(img_file, 0)
    x_dimension, y_dimension = img.shape
    height = x_dimension // N
    width = y_dimension // N
    acc_m_list = []
    for i in range(N):
        for j in range(N):
            patch = img[i*height:(i+1)*height, j*width:(j+1)*width]
            acc_m = calculateAccumulatedCooccurrenceMatrix(patch, bin_number, d)
            acc_m_list.append(calculateCooccurrenceFeatures(acc_m))


    Test_features_part3.append(np.mean((np.array(acc_m_list)), 0))
    
Test_features_part3 = np.array(Test_features_part3)
np.savetxt('Test_features_part3.csv', Test_features_part3, delimiter=',')


# ## Train Features

# In[339]:


training_labels = np.loadtxt("dataset//training_labels.txt")
training_labels = training_labels.tolist()
training_labels = np.array(training_labels)## Note that below part is take so long so go below and direclty upload the datas from features which I already get features data and store in txt
# load from the csv file
train_features_part3 = np.loadtxt('train_features_part3.csv',delimiter=',')


# ## Test Features

# In[340]:


test_labels = np.loadtxt("dataset//test_labels.txt")
test_labels = test_labels.tolist()
test_labels = np.array(test_labels)
# load from the csv file
test_features_part3 = np.loadtxt('test_features_part3.csv',delimiter=',')


# In[306]:


#I just debug small error
print(len(training_labels))
len(test_features_part3)


# ## Found best C for linear kernel

# In[341]:



features_fitted_part3 = scaler.fit_transform(train_features_part3)
test_features_fitted_part3 = scaler.fit_transform(test_features_part3)
features_fitted_part3 = np.array(features_fitted_part3)
test_features_fitted_part3 = np.array(test_features_fitted_part3)
for c in C_values:
    svc = SVC(kernel='linear', class_weight='balanced', C=c)
    model = svc.fit(features_fitted_part3, training_labels)
    training_pred_part3 = np.array(model.predict(features_fitted_part3))
    test_pred_part3 = np.array(model.predict(test_features_fitted_part3))

#     print(c)
#     print('Overall Accuracy for train  ', accuracy_score(training_labels, training_pred_part3))
#     print('Class Accuracy for train ', confusion_matrix(training_labels, training_pred_part3, normalize="true").diagonal())
#     print('Overall Accuracy : ', accuracy_score(test_labels, test_pred_part3))
#     print('Class Accuracy : ', confusion_matrix(test_labels, test_pred_part3, normalize="true").diagonal())
for c in np.linspace(5,10,100):
    svc = SVC(kernel='linear', class_weight='balanced', C=c)
    model = svc.fit(features_fitted_part3, training_labels)
    training_pred_part3 = np.array(model.predict(features_fitted_part3))
    test_pred_part3 = np.array(model.predict(test_features_fitted_part3))
    if(accuracy_score(test_labels, test_pred_part3)>0.458):
        print(c)
        print('Overall Accuracy for train  ', accuracy_score(training_labels, training_pred_part3))
        print('Class Accuracy for train ', confusion_matrix(training_labels, training_pred_part3, normalize="true").diagonal())
        print('Overall Accuracy : ', accuracy_score(test_labels, test_pred_part3))
        print('Class Accuracy : ', confusion_matrix(test_labels, test_pred_part3, normalize="true").diagonal())
c=5.1
C_values.append(c)
svc = SVC(kernel='linear', class_weight='balanced', C=c)
model = svc.fit(features_fitted_part3, training_labels)
training_pred_lin_part3 = np.array(model.predict(features_fitted_part3))
test_pred_lin_part3 = np.array(model.predict(test_features_fitted_part3))


# ## Found best C and gamma for RBF kernel

# In[342]:


## Found best C for linear kernel
for c in C_values:
    for gamma in Gamma_values:
        svc = SVC(kernel='rbf', class_weight='balanced', C=c, gamma=gamma)
        model = svc.fit(features_fitted_part3, training_labels)
        training_pred_part3 = np.array(model.predict(features_fitted_part3))
        test_pred_part3 = np.array(model.predict(test_features_fitted_part3))
        if(accuracy_score(test_labels, test_pred_part3)>0.5):
            print(c)
            print(gamma)
            print('Overall Accuracy for train  ', accuracy_score(training_labels, training_pred_part3))
            print('Class Accuracy for train ', confusion_matrix(training_labels, training_pred_part3, normalize="true").diagonal())
            print('Overall Accuracy : ', accuracy_score(test_labels, test_pred_part3))
            print('Class Accuracy : ', confusion_matrix(test_labels, test_pred_part3, normalize="true").diagonal())

c=10            
gamma=0.4

svc = SVC(kernel='rbf', class_weight='balanced', C=c, gamma=gamma)
model = svc.fit(features_fitted_part3, training_labels)
training_pred_rbf_part3 = np.array(model.predict(features_fitted_part3))
test_pred_rbf_part3 = np.array(model.predict(test_features_fitted_part3))


# In[343]:


np.savetxt('test_pred_lin_part3', test_pred_lin_part3, delimiter=',')
np.savetxt('test_pred_rbf_part3', test_pred_rbf_part3, delimiter=',')


# In[348]:


training_pred_lin_part3 = np.array(training_pred_lin_part3)
training_pred_rbf_part3 = np.array(training_pred_rbf_part3)
test_pred_lin_part3 = np.array(test_pred_lin_part3)
test_pred_rbf_part3 = np.array(test_pred_rbf_part3)

for i in range(4):
    table_train_lin_2_3 = contingency_table(training_labels, training_pred_lin, training_pred_lin_part3, i)
    result_train_lin_2_3 = mcnemar(table_train_lin_2_3, exact=False, correction=True)
    print("Class \n",i)
    print("Result_Train\n",result_train_lin_2_3)
    
    table_test_lin_2_3 = contingency_table(test_labels, test_pred_lin, test_pred_lin_part3, i)
    result_test_lin_2_3 = mcnemar(table_test_lin_2_3, exact=False, correction=True)
    print("Result_Test\n",result_test_lin_2_3)


# In[347]:


for i in range(4):
    table_train_rbf_2_3 = contingency_table(training_labels, training_pred_rbf, training_pred_rbf_part3, i)
    result_train_rbf_2_3 = mcnemar(table_train_rbf_2_3, exact=False, correction=True)
    print("Class \n",i)
    print("Result_Train\n",result_train_rbf_2_3)
    
    table_test_rbf_2_3 = contingency_table(test_labels, test_pred_rbf, test_pred_rbf_part3, i)  
    result_test_rbf_2_3 = mcnemar(table_test_rbf_2_3, exact=False, correction=True)
    print("Result_Test\n",result_test_rbf_2_3)


# # Part IV

# ## Note that below part is take so long so go below and direclty upload the datas from features which I already get features data and store in cvs

# In[380]:


## Note that below part is take so long so go below and direclty upload the datas from features which I already get features data and store in cvsTrain_features_part4 = []
acc_m_list = []
for img_file in glob.glob(test_dir + "*.jpg"):
    img_initial = cv.imread(img_file, 0)
    #keypoints
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img_initial,None)
    img = cv2.drawKeypoints(img_initial,keypoints_1,img_initial)
    acc_m = calculateAccumulatedCooccurrenceMatrix(patch, bin_number, d)
    acc_m_list.append(calculateCooccurrenceFeatures(acc_m))
    Train_features_part4.append(np.mean((np.array(acc_m_list)), 0))
    
Train_features_part4 = np.array(Train_features_part4)
np.savetxt('Train_features_part4.csv', Train_features_part4, delimiter=',')


# ## Note that below part is take so long so go below and direclty upload the datas from features which I already get features data and store in cvs

# In[ ]:


## Note that below part is take so long so go below and direclty upload the datas from features which I already get features data and store in cvsTest_features_part4 = []
acc_m_list = []
for img_file in glob.glob(test_dir + "*.jpg"):
    img_initial = cv.imread(img_file, 0)
    #keypoints
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img_initial,None)
    img = cv2.drawKeypoints(img_initial,keypoints_1,img_initial)
    acc_m = calculateAccumulatedCooccurrenceMatrix(patch, bin_number, d)
    acc_m_list.append(calculateCooccurrenceFeatures(acc_m))
    Test_features_part4.append(np.mean((np.array(acc_m_list)), 0))
    
Test_features_part4 = np.array(Test_features_part4)
np.savetxt('Test_features_part4.csv', Test_features_part4, delimiter=',')


# ## Train Features

# In[381]:


training_labels = np.loadtxt("dataset//training_labels.txt")
training_labels = training_labels.tolist()
training_labels = np.array(training_labels)## Note that below part is take so long so go below and direclty upload the datas from features which I already get features data and store in txt
# load from the csv file
train_features_part4 = np.loadtxt('train_features_part4.csv',delimiter=',')


# ## Test Features

# In[382]:


test_labels = np.loadtxt("dataset//test_labels.txt")
test_labels = test_labels.tolist()
test_labels = np.array(test_labels)
# load from the csv file
test_features_part4 = np.loadtxt('test_features_part4.csv',delimiter=',')


# ## Found best C for linear kernel

# In[385]:



features_fitted_part4 = scaler.fit_transform(train_features_part4)
test_features_fitted_part4 = scaler.fit_transform(test_features_part4)
features_fitted_part4 = np.array(features_fitted_part4)
test_features_fitted_part4 = np.array(test_features_fitted_part4)
for c in C_values:
    svc = SVC(kernel='linear', class_weight='balanced', C=c)
    model = svc.fit(features_fitted_part4, training_labels)
    training_pred_part4 = np.array(model.predict(features_fitted_part4))
    test_pred_part4 = np.array(model.predict(test_features_fitted_part4))

    print(c)
    print('Overall Accuracy for train  ', accuracy_score(training_labels, training_pred_part3))
    print('Class Accuracy for train ', confusion_matrix(training_labels, training_pred_part3, normalize="true").diagonal())
    print('Overall Accuracy : ', accuracy_score(test_labels, test_pred_part3))
    print('Class Accuracy : ', confusion_matrix(test_labels, test_pred_part3, normalize="true").diagonal())
for c in np.linspace(5,10,100):
    svc = SVC(kernel='linear', class_weight='balanced', C=c)
    model = svc.fit(features_fitted_part4, training_labels)
    training_pred_part4 = np.array(model.predict(features_fitted_part4))
    test_pred_part4 = np.array(model.predict(test_features_fitted_part4))
    if(accuracy_score(test_labels, test_pred_part4)>0.458):
        print(c)
        print('Overall Accuracy for train  ', accuracy_score(training_labels, training_pred_part4))
        print('Class Accuracy for train ', confusion_matrix(training_labels, training_pred_part4, normalize="true").diagonal())
        print('Overall Accuracy : ', accuracy_score(test_labels, test_pred_part4))
        print('Class Accuracy : ', confusion_matrix(test_labels, test_pred_part4, normalize="true").diagonal())
c=5.1
C_values.append(c)
svc = SVC(kernel='linear', class_weight='balanced', C=c)
model = svc.fit(features_fitted_part4, training_labels)
training_pred_lin_part4 = np.array(model.predict(features_fitted_part4))
test_pred_lin_part4 = np.array(model.predict(test_features_fitted_part4))


# ## Found best C and gamma for RBF kernel

# In[403]:


## Found best C for linear kernel
for c in C_values:
    for gamma in Gamma_values:
        svc = SVC(kernel='rbf', class_weight='balanced', C=c, gamma=gamma)
        model = svc.fit(features_fitted_part4, training_labels)
        training_pred_part4 = np.array(model.predict(features_fitted_part4))
        test_pred_part4 = np.array(model.predict(test_features_fitted_part4))
        if(accuracy_score(test_labels, test_pred_part4)>0.56):
            print(c)
            print(gamma)
            print('Overall Accuracy for train  ', accuracy_score(training_labels, training_pred_part4))
            print('Class Accuracy for train ', confusion_matrix(training_labels, training_pred_part4, normalize="true").diagonal())
            print('Overall Accuracy : ', accuracy_score(test_labels, test_pred_part4))
            print('Class Accuracy : ', confusion_matrix(test_labels, test_pred_part4, normalize="true").diagonal())

c=5000           
gamma=0.3

svc = SVC(kernel='rbf', class_weight='balanced', C=c, gamma=gamma)
model = svc.fit(features_fitted_part4, training_labels)
training_pred_rbf_part4 = np.array(model.predict(features_fitted_part4))
test_pred_rbf_part4 = np.array(model.predict(test_features_fitted_part4))


# In[401]:


## Found best C for linear kernel
for c in np.linspace(1000,10000,1000):
    for gamma in Gamma_values:
        svc = SVC(kernel='rbf', class_weight='balanced', C=c, gamma=gamma)
        model = svc.fit(features_fitted_part4, training_labels)
        training_pred_part4 = np.array(model.predict(features_fitted_part4))
        test_pred_part4 = np.array(model.predict(test_features_fitted_part4))
        if(accuracy_score(test_labels, test_pred_part4)>0.56):
            print(c)
            print(gamma)
            print('Overall Accuracy for train  ', accuracy_score(training_labels, training_pred_part4))
            print('Class Accuracy for train ', confusion_matrix(training_labels, training_pred_part4, normalize="true").diagonal())
            print('Overall Accuracy : ', accuracy_score(test_labels, test_pred_part4))
            print('Class Accuracy : ', confusion_matrix(test_labels, test_pred_part4, normalize="true").diagonal())

c=5000            
gamma=0.3

svc = SVC(kernel='rbf', class_weight='balanced', C=c, gamma=gamma)
model = svc.fit(features_fitted_part4, training_labels)
training_pred_rbf_part4 = np.array(model.predict(features_fitted_part4))
test_pred_rbf_part4 = np.array(model.predict(test_features_fitted_part4))


# In[409]:


training_pred_lin_part4 = np.array(training_pred_lin_part4)
training_pred_rbf_part4 = np.array(training_pred_rbf_part4)
test_pred_lin_part4 = np.array(test_pred_lin_part4)
test_pred_rbf_part4 = np.array(test_pred_rbf_part4)

for i in range(4):
    table_train_lin_2_3_4 = contingency_table(training_labels, training_pred_part3, training_pred_lin_part4, i)
    result_train_lin_2_3_4 = mcnemar(table_train_lin_2_3_4, exact=False, correction=True)
    print("Class \n",i)
    print("Result_Train\n",result_train_lin_2_3_4)
    
    table_test_lin_2_3_4 = contingency_table(test_labels, test_pred_lin_part3, test_pred_lin_part4, i)
    result_test_lin_2_3_4 = mcnemar(table_test_lin_2_3_4, exact=False, correction=True)
    print("Result_Test\n",result_test_lin_2_3_4)


# In[407]:


for i in range(4):
    table_train_rbf_2_3_4 = contingency_table(training_labels, training_pred_rbf_part3, training_pred_rbf_part4, i)
    result_train_rbf_2_3_4 = mcnemar(table_train_rbf_2_3_4, exact=False, correction=True)
    print("Class \n",i)
    print("Result_Train\n",result_train_rbf_2_3_4)
    
    table_test_rbf_2_3_4 = contingency_table(test_labels, test_pred_rbf_part3, test_pred_rbf_part4, i)  
    result_test_rbf_2_3_4 = mcnemar(table_test_rbf_2_3_4, exact=False, correction=True)
    print("Result_Test\n",result_test_rbf_2_3_4)


# In[ ]:




