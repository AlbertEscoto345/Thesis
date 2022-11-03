import numpy as np 
import cv2
import glob
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix,classification_report
from mlxtend.plotting import plot_confusion_matrix

print(os.listdir("C:/thesis dataset/dataset_troubleshoot 2/"))
dim = 50

def getYourFruits(fruits, data_type, print_n=False, k_fold=False):
    images = []
    labels = []
    val = ['Training', 'Test']
    if not k_fold:
        path = "C:/thesis dataset/dataset_troubleshoot 2/*/fruits-360/" + data_type + "/"
        for i,f in enumerate(fruits):
            p = path + f
            j=0
            for image_path in glob.glob(os.path.join(p, "*.jpg")):
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (50,50))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                images.append(image)
                labels.append(i)
                j+=1
            if(print_n):
                print("There are " , j , " " , data_type.upper(), " images of " , fruits[i].upper())
        images = np.array(images)
        labels = np.array(labels)
        return images, labels
    else:
        for v in val:
            path = "C:/thesis dataset/dataset_troubleshoot 2/*/fruits-360/" + v + "/"
            for i,f in enumerate(fruits):
                p = path + f
                j=0
                for image_path in glob.glob(os.path.join(p,"*.jpg")):
                    image = cv2.imread(image_path,cv2.IMREAD_COLOR)
                    image = cv2.resize(image,(50,50))
                    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                    images.append(image)
                    labels.append(i)
                    j+=1
        images = np.array(images)
        labels = np.array(labels)
        return images, labels
    
def getAllFruits():
    fruits = []
    for fruit_path in glob.glob("C:/thesis dataset/dataset_troubleshoot 2/*/fruits-360/Training/*"):
        fruit = fruit_path.split("/")[-1]
        fruits.append(fruit)
    return fruits

#Choose your Fruits
fruits = ['Cool-colored' , 'Warm-colored'] #Binary classification

#Get Images and Labels 
X_t, y_train =  getYourFruits(fruits, 'Training', print_n=True, k_fold=False)
X_test, y_test = getYourFruits(fruits, 'Test', print_n=True, k_fold=False)

#Get data for k-fold
X,y = getYourFruits(fruits, '', print_n=True, k_fold=True)

#Scale Data Images
scaler = StandardScaler()
X_train = scaler.fit_transform([i.flatten() for i in X_t])
X_test = scaler.fit_transform([i.flatten() for i in X_test])
X = scaler.fit_transform([i.flatten() for i in X])

testscoreS=[]

#KNN3
knn = KNeighborsClassifier(n_neighbors=3,p=2,metric='euclidean')
knn.fit(X_train, y_train)
trainscore = knn.score(X_train,y_train)
testscore = knn.score(X_test,y_test)
testscoreS.append(testscore)
y_pred = knn.predict(X_test)
#Evaluation
precision = metrics.accuracy_score(y_pred, y_test) * 100
print("Accuracy with K-NN: {0:.2f}%".format(precision))
print("CLASSIFICATION REPORT FOR KNN")
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print()
print(classification_report(y_test,y_pred,target_names=fruits))
print("train score:",trainscore)
print("test_score",testscore)
## Display the visualization of the Confusion Matrix.
fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test,y_pred),figsize=(6, 6),cmap=plt.cm.Greens,class_names=fruits)
plt.xlabel('Predicted Class',fontsize=18)
plt.ylabel('Actual Class',fontsize=18)
plt.title('KNN Confusion Matrix',fontsize=18)
plt.show()

#KNN4
knn = KNeighborsClassifier(n_neighbors=4,p=2,metric='euclidean')
knn.fit(X_train, y_train)
trainscore = knn.score(X_train,y_train)
testscore = knn.score(X_test,y_test)
testscoreS.append(testscore)
y_pred = knn.predict(X_test)
#Evaluation
precision = metrics.accuracy_score(y_pred, y_test) * 100
print("Accuracy with K-NN: {0:.2f}%".format(precision))
print("CLASSIFICATION REPORT FOR KNN")
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print()
print(classification_report(y_test,y_pred,target_names=fruits))
print("train score:",trainscore)
print("test_score",testscore)
## Display the visualization of the Confusion Matrix.
fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test,y_pred),figsize=(6, 6),cmap=plt.cm.Greens,class_names=fruits)
plt.xlabel('Predicted Class',fontsize=18)
plt.ylabel('Actual Class',fontsize=18)
plt.title('KNN Confusion Matrix',fontsize=18)
plt.show()

#KNN5
knn = KNeighborsClassifier(p=2,metric='euclidean')
knn.fit(X_train, y_train)
trainscore = knn.score(X_train,y_train)
testscore = knn.score(X_test,y_test)
testscoreS.append(testscore)
y_pred = knn.predict(X_test)
#Evaluation
precision = metrics.accuracy_score(y_pred, y_test) * 100
print("Accuracy with K-NN: {0:.2f}%".format(precision))
print("CLASSIFICATION REPORT FOR KNN")
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print()
print(classification_report(y_test,y_pred,target_names=fruits))
print("train score:",trainscore)
print("test_score",testscore)
## Display the visualization of the Confusion Matrix.
fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test,y_pred),figsize=(6, 6),cmap=plt.cm.Greens,class_names=fruits)
plt.xlabel('Predicted Class',fontsize=18)
plt.ylabel('Actual Class',fontsize=18)
plt.title('KNN Confusion Matrix',fontsize=18)
plt.show()

#KNN6
knn = KNeighborsClassifier(n_neighbors=6,p=2,metric='euclidean')
knn.fit(X_train, y_train)
trainscore = knn.score(X_train,y_train)
testscore = knn.score(X_test,y_test)
testscoreS.append(testscore)
y_pred = knn.predict(X_test)
#Evaluation
precision = metrics.accuracy_score(y_pred, y_test) * 100
print("Accuracy with K-NN: {0:.2f}%".format(precision))
print("CLASSIFICATION REPORT FOR KNN")
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print()
print(classification_report(y_test,y_pred,target_names=fruits))
print("train score:",trainscore)
print("test_score",testscore)
## Display the visualization of the Confusion Matrix.
fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test,y_pred),figsize=(6, 6),cmap=plt.cm.Greens,class_names=fruits)
plt.xlabel('Predicted Class',fontsize=18)
plt.ylabel('Actual Class',fontsize=18)
plt.title('KNN Confusion Matrix',fontsize=18)
plt.show()

#KNN7
knn = KNeighborsClassifier(n_neighbors=7,p=2,metric='euclidean')
knn.fit(X_train, y_train)
trainscore = knn.score(X_train,y_train)
testscore = knn.score(X_test,y_test)
testscoreS.append(testscore)
y_pred = knn.predict(X_test)
#Evaluation
precision = metrics.accuracy_score(y_pred, y_test) * 100
print("Accuracy with K-NN: {0:.2f}%".format(precision))
print("CLASSIFICATION REPORT FOR KNN")
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print()
print(classification_report(y_test,y_pred,target_names=fruits))
print("train score:",trainscore)
print("test_score",testscore)
## Display the visualization of the Confusion Matrix.
fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test,y_pred),figsize=(6, 6),cmap=plt.cm.Greens,class_names=fruits)
plt.xlabel('Predicted Class',fontsize=18)
plt.ylabel('Actual Class',fontsize=18)
plt.title('KNN Confusion Matrix',fontsize=18)
plt.show()

#Relationship between K and Testing Accuracy
k_range=range(3,8,1)
plt.plot(k_range,testscoreS)
plt.locator_params(axis='x', nbins=5)
plt.xlabel('Values of K',fontsize=18)
plt.ylabel('Testing Accuracy',fontsize=18)