import numpy as np
import pandas as pd
from sklearn import svm
import glob
from sklearn.model_selection import train_test_split
import csv
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

'''model= svm.SVC(kernel='rbf', C = 20 , gamma='scale')  0.92, 0.85'''
'''model= svm.SVC(kernel='rbf', C = 10 , gamma='scale')  0.885 0.84'''
'''model= svm.SVC(kernel='rbf', C = 15 , gamma='scale')   0.90 0.86'''
'''model= svm.SVC(kernel='linear', C = 10 , gamma='scale')   0.99 , 0.787'''
'''model= svm.SVC(kernel='linear', C = 5 , gamma='scale')   0.99 , 0.80'''
'''model= svm.SVC(kernel='linear', C = 10 , gamma='auto')   0.99 0.92'''
'''model= svm.SVC(kernel='linear', C = 15 , gamma='auto')   0.99 0.936-----upload site: 0.825 '''

'''poly kernel:   c=15 gamma=scale  1.000,0.88'''
'''poly kernel:     d=7 c=5 gamma =scale 1.000 0.909   '''
'''poly kernel:     d=7 c=5 gamma =auto   1.000 0.90 '''
'''poly kernel:     d=7 c=10 gamma =scale 1.000 0.89   '''
'''poly kernel:     d=7 c=10 gamma =auto   1.000 0.90 '''

'''poly kernel:     d=10 c=20 gamma =scale   1.000 0.919 '''
'''poly kernel:     d=15 c=15 gamma =auto   1.000 0.920 '''
'''poly kernel:     d=15 c=25 gamma =auto   1.000 0.927 -------- 0.819'''

'''model= svm.SVC(kernel='rbf' , C = 0.5 ,gamma='auto', decision_function_shape='ovo' )'''

'''model= svm.SVC(kernel='rbf' , C = 4 ,gamma='auto')    0.999,0.93  ------  0.816'''
'''model= svm.SVC(kernel='rbf' , C = 5 ,gamma='auto')       -------  '''
'''model= svm.SVC(kernel='rbf' , C = 6 ,gamma='auto')  0.99, 0.94  splituire 0.9,0.1  '''
'''model= svm.SVC(kernel='rbf' , C = 6 ,gamma='auto')  0.97, 0.92  splituire 0.9,0.1  -------0.802'''

'''model= svm.SVC(kernel='rbf' , C = 5 ,gamma='auto')    0.99, 0.938   -------  '''

'''model= svm.SVC(kernel='rbf' , C = 5 ,gamma='auto', decision_function_shape='ovo') split 0.9,0.1   0.994, 0.951   ------- 0.829   maine fa cu splituire normala'''

'''model= svm.SVC(kernel='rbf' , C = 10 ,gamma='auto') -------  8.30'''

'''model= svm.SVC(kernel='rbf'  , C = 7 ,gamma='auto')   0.9/0.1 split ------- 8.27'''

path_train = 'D:/Proiecte/Kaggle_ML/train'
path_label = 'D:/Proiecte/Kaggle_ML/train_labels.csv'
path_test = 'D:/Proiecte/Kaggle_ML/test'


all_files = glob.glob(path_train + "/*.csv")
all_files_test= glob.glob(path_test + "/*.csv")


labels = pd.read_csv(path_label, usecols=[1],header=0)
labels = np.array(labels).astype(float)

Y_train = np.array(labels)



X_test=[]
X_train =[]
ID = []



for filename in all_files:
    date = pd.read_csv(filename, index_col=None, header=None)
    date = date.values.tolist()

    print(filename)  #incarcam datele din train data in X_train
    if(len(date)<150):
        while(len(date) < 150):
            date.append([0.0,0.0,0.0])
    if(len(date)>150):
        date = date[0:150];
    date = np.array(date).astype(float)
    X_train.append(date)

for filename in all_files_test: #la fel si pt X_test
    date = pd.read_csv(filename, index_col=None, header=None)
    date = date.values.tolist()

    ID.append(filename[41:46])  #vreau sa iau doar numerele din fisiere(id-ul)
    print(filename[41:46])


    if(len(date)<150):
        while(len(date) < 150):
            date.append([0.0,0.0,0.0])
    if(len(date)>150):
        date = date[0:150];
    date = np.array(date).astype(float)
    X_test.append(date)


#transf datele in np.array

X_train = np.array(X_train)
X_test = np.array(X_test)

print(X_train.shape)
print(X_test.shape)

#
X_train = X_train.reshape(9000,450)
X_test = X_test.reshape(5000,450)

X_trained,X_tested,Y_trained,Y_tested = train_test_split(X_train, Y_train, train_size=0.9, test_size=0.1)  #splituiesc informatiile fie 90/10 sau 80/20
model = svm.SVC(kernel='rbf'  , C = 7 ,gamma='auto')

Y_trained = Y_trained.reshape(8100,) #il trans in vect linie,8100 daca e 90/10 si 7200 daca e 80/20
model.fit(X_trained,Y_trained)

print("Train accuracy model: %f" % model.score(X_trained,Y_trained)) #train score
print("Test accuracy model: %f" % model.score(X_tested,Y_tested))  #test score


Y_test = []
Y_test = model.predict(X_test)


score = []
Y_pred  =[]
Y_pred = model.predict(X_train)
cv = KFold(n_splits = 3 , random_state=42,shuffle= False)  #functia 3-kfold: parametrii random_state si shuffle nu s necesari in acest caz

for train_index, test_index in cv.split(X_train):

    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)
    X_tr, X_te, Y_tr, Y_te = X_train[train_index], X_train[test_index], Y_train[train_index], Y_train[test_index]
    model.fit(X_tr,Y_tr)
    score.append(model.score(X_te,Y_te))
    print("3-Fold Score : %f" % np.mean(score))
    cm = confusion_matrix(Y_train, Y_pred)
    print(cm)



with open('D:/Python/Python_projects/Kaggle_ML/Results.csv','r+', newline='') as csv_file:  #scrierea in fisier a rezultatelor
    writer = csv.writer(csv_file)
    writer.writerow(['id','class'])
    for i in range(len(Y_test)):
        writer.writerow([int(ID[i]),int(Y_test[i])])



