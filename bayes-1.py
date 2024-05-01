from __future__ import division
import math
from sklearn import metrics,preprocessing
from sklearn.metrics import classification_report,precision_recall_fscore_support,accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
import numpy

spam_data=shuffle(numpy.loadtxt('spambase.data', delimiter=','))
features_unscaled=spam_data[:,:-1]
features=preprocessing.StandardScaler().fit_transform(features_unscaled)
labels=spam_data[:,57]
#split the data into test and train set
train_set, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.50)

#train data with spam and not spam
print("Train set:")
not_spam_train = numpy.nonzero(labels_train==0)[0]
spam_train = numpy.nonzero(labels_train==1)[0]
print("Not-spam instances: ",len(not_spam_train))
print("Spam Instances: ",len(spam_train))
#test data with spam and not spam
print("\nTest set:")
non_spam_test = numpy.nonzero(labels_test==0)[0]
spam_test = numpy.nonzero(labels_test==1)[0]
print("Not-spam instances: ",len(non_spam_test))
print("Spam Instances: ",len(spam_test))

#Computing prior probability
spam_prob=int(len(spam_train))/(int(len(spam_train))+int(len(not_spam_train)))
non_spam_prob=int(len(not_spam_train))/(int(len(spam_train))+int(len(not_spam_train)))
logPclass = numpy.log([non_spam_prob,spam_prob])

indices=[numpy.nonzero(labels_train==0)[0],numpy.nonzero(labels_train)[0]]
#computing mean and standard deviation
mean=numpy.transpose([numpy.mean(train_set[indices[0],:],axis=0),numpy.mean(train_set[indices[1],:],axis=0)])
std=numpy.transpose([numpy.std(train_set[indices[0],:],axis=0),numpy.std(train_set[indices[1],:],axis=0)])
#replace zero standard deviation with small standard deviation
zero_std = numpy.nonzero(std==0)[0]
if (numpy.any(zero_std)):
    numpy.place(std,std==0,0.0001)

predictions=[]
#naive bayes classification on test set
for i in range(0, features_test.shape[0]):
    denom=math.sqrt(2*numpy.pi)*std
    index=-1*(numpy.divide(numpy.power(numpy.subtract(features_test[i,:].reshape(features_test.shape[1],1),mean), 2),2*numpy.power(std, 2)))
    num=numpy.exp(index)
    zero_num = numpy.nonzero(num==0)[0]
    if (numpy.any(zero_num)):
        numpy.place(num,num==0,0.1e-250)
    pdf = numpy.divide(num,denom)
    #class prediction for test example
    prediction = numpy.argmax(logPclass+numpy.sum(numpy.nan_to_num(numpy.log(pdf)), axis=0))
    predictions.append(prediction)

#calculate accuracy,recall,precision and confusion matrix
acc=accuracy_score(predictions,labels_test)
print("\nClassification Report:")
print(classification_report(labels_test,predictions))
print(f"\nAccuracy: {acc}")
recall = precision_recall_fscore_support(labels_test, predictions)[1]
print("Recall:", recall)
precision= precision_recall_fscore_support(labels_test, predictions)[0]
print("Precision:", precision)
print("\nConfusion matrix on test set:")
print(confusion_matrix(labels_test,predictions))