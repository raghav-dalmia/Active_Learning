'''Importing the libraries '''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math

''' Importing the dataset '''
dataset = pd.read_csv('tic-tac-toe.data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,9].values

'''Encoding the categorical data '''
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
for i in range(0,9):
  labelencoder_x_i = LabelEncoder()
  X[:,i] = labelencoder_x_i.fit_transform(X[:,i])
# Label Encoded into 0,1,2
# One Hot Encoding of Independent features (Obtained 27 features)
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
#Label Encoding the Dependent Variable
labelencoder_y =  LabelEncoder()
y = labelencoder_y.fit_transform(y)
# Removing the dummy variable trap (18 features left)
l=[]
for i in range(27):
  if (i+1)%3!=0:
    l.append(i)
X = X[:, l]

'''Keeping the Mapping of X and y'''
y = y.reshape(-1,1)
new_df = np.concatenate((X,y),axis=1)

'''Ensemble Sampling '''
# Taking Samples in the ratio of 1:1 
df1 = new_df[new_df[:,18] == 1]
df2 = new_df[new_df[:,18] == 0]
np.random.shuffle(df1) # All +ve values
np.random.shuffle(df2) # All -ve values

'''Random Sampling'''
np.random.shuffle(new_df)
L = new_df[:250,:]
L_X = L[:,:-1]
L_y = L[:,-1]

# Splitting the dataset into Labeled and Unlabeled dataset (ratio 1:1)
train_size = 50
size = int(train_size/2)
labeled_dataset = np.concatenate((df1[0:size,:] , df2[0:size,:]), axis = 0)
unlabeled_dataset = np.concatenate((df1[size+1 :,:] , df2[size+1 :,:]) , axis = 0)

# Splitting the dataset into training set and test set
X_train = labeled_dataset[:,:-1]
y_train = labeled_dataset[:,-1]
test_size = 100
X_test = unlabeled_dataset[:test_size , :-1]
y_test = unlabeled_dataset[:test_size ,-1]
unlabeled_dataset = unlabeled_dataset[test_size:,:]
X_unlabel = unlabeled_dataset[:,:-1]
y_unlabel = unlabeled_dataset[:,-1]

'''Train Random Sampling Model'''
from sklearn.naive_bayes import GaussianNB
classifier1 = GaussianNB()
classifier1.fit(L_X,L_y)
y_pred1 = classifier1.predict(L_X)
random_accuracy = classifier1.score(L_X,L_y)

'''Train the Model using Naive Bayes Classifier'''
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

'''Predicting the test set results'''
y_pred = classifier.predict(X_test)

'''Making the Confusion Matrix'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

'''Predicting accuracy'''
new_accuracy = classifier.score(X_test,y_test)
print("Initial Accuracy : {}".format(new_accuracy))

'''Query By committee Strategy'''
# Processing committee members using k-means clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6, init='k-means++',max_iter = 300, n_init = 10 , random_state = 0)
y_means = kmeans.fit_predict(X_train)

# Prepare two models on the basis of clusters
cluster_1 = []
cluster_2 = []
cluster_3 = []
cluster_4 = []
cluster_5 = []
cluster_6 = []
for i in range(0,len(y_means)):
  if y_means[i] == 0:
    cluster_1.append(labeled_dataset[i,:])
  elif y_means[i] == 1:
    cluster_2.append(labeled_dataset[i,:])
  elif y_means[i] == 2:
    cluster_3.append(labeled_dataset[i,:])
  elif y_means[i] == 3:
    cluster_4.append(labeled_dataset[i,:])
  elif y_means[i] == 4:
    cluster_5.append(labeled_dataset[i,:])
  else:
    cluster_6.append(labeled_dataset[i,:])

cluster_1 = np.array(cluster_1)
cluster_2 = np.array(cluster_2)
cluster_3 = np.array(cluster_3)
cluster_4 = np.array(cluster_4)
cluster_5 = np.array(cluster_5)
cluster_6 = np.array(cluster_6)
print(cluster_1.shape)
print(cluster_2.shape)
print(cluster_3.shape)
print(cluster_4.shape)
print(cluster_5.shape)
print(cluster_5.shape)

# Cluster 1 model M1
cluster_1_X = cluster_1[:,:-1]
cluster_1_y = cluster_1[:,-1]
cluster_1_classifier = GaussianNB()
cluster_1_classifier.fit(cluster_1_X,cluster_1_y)
# Cluster 2 model M2
cluster_2_X = cluster_2[:,:-1]
cluster_2_y = cluster_2[:,-1]
cluster_2_classifier = GaussianNB()
cluster_2_classifier.fit(cluster_2_X,cluster_2_y)
# Cluster 3 model M3
cluster_3_X = cluster_3[:,:-1]
cluster_3_y = cluster_3[:,-1]
cluster_3_classifier = GaussianNB()
cluster_3_classifier.fit(cluster_3_X,cluster_3_y)
# Cluster 4 model M4
cluster_4_X = cluster_4[:,:-1]
cluster_4_y = cluster_4[:,-1]
cluster_4_classifier = GaussianNB()
cluster_4_classifier.fit(cluster_4_X,cluster_4_y)
# Cluster 5 model M5
cluster_5_X = cluster_5[:,:-1]
cluster_5_y = cluster_5[:,-1]
cluster_5_classifier = GaussianNB()
cluster_5_classifier.fit(cluster_5_X,cluster_5_y)
# Cluster 6 model M6
cluster_6_X = cluster_6[:,:-1]
cluster_6_y = cluster_6[:,-1]
cluster_6_classifier = GaussianNB()
cluster_6_classifier.fit(cluster_6_X,cluster_6_y)


# Picking up queries
no_of_queries = 150 
query_count = 150
mean = 0
accuracies = []
while (no_of_queries != 0):
  number = random.randint(0,unlabeled_dataset.shape[0]-1)
  sample = unlabeled_dataset[number,:-1][np.newaxis,:]
  cluster_1_y_pred = cluster_1_classifier.predict(sample) #class label
  cluster_2_y_pred = cluster_2_classifier.predict(sample) #class label
  cluster_3_y_pred = cluster_3_classifier.predict(sample) #class label
  cluster_4_y_pred = cluster_4_classifier.predict(sample) #class label
  cluster_5_y_pred = cluster_5_classifier.predict(sample) #class label
  cluster_6_y_pred = cluster_6_classifier.predict(sample) #class label

  positive = 0
  negative = 0
  if cluster_1_y_pred == 1 :
    positive = positive + 1
  else:
    negative = negative + 1
  if cluster_2_y_pred == 1 :
    positive = positive + 1
  else:
    negative = negative + 1
  if cluster_3_y_pred == 1 :
    positive = positive + 1
  else:
    negative = negative + 1
  if cluster_4_y_pred == 1 :
    positive = positive + 1
  else:
    negative = negative + 1
  if cluster_5_y_pred == 1 :
    positive = positive + 1
  else:
    negative = negative + 1
  if cluster_6_y_pred == 1 :
    positive = positive + 1
  else:
    negative = negative + 1

  print("No of positive values : {}" .format(positive))
  print("No of negative values : {}" .format(negative))

  # Calculating vote entropy for the sample
  vote_entropy = 0
  if positive != 0:
    vote_entropy = vote_entropy + (-1 * (positive/5) * math.log(positive/5.0))
  if negative != 0:
    vote_entropy = vote_entropy + (-1 * (negative/5) * math.log(negative/5.0)) 
  print("Vote Entropy : {}" .format(vote_entropy))
  if vote_entropy > 0.5 :
    new_sample = unlabeled_dataset[number,:][np.newaxis,:]
    labeled_dataset = np.concatenate((labeled_dataset ,new_sample),axis=0)
    print(labeled_dataset.shape)
    unlabeled_dataset = np.delete(unlabeled_dataset,(number),axis=0)
    print(unlabeled_dataset.shape)
    no_of_queries = no_of_queries - 1

    X_train = labeled_dataset[:,:-1]
    y_train = labeled_dataset[:,-1]

    '''Train the Model using Naive Bayes Classifier'''
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train,y_train)

    '''Predicting the test set results'''
    y_pred = classifier.predict(X_test)

    '''Predicting accuracy'''
    new_accuracy = classifier.score(X_test,y_test)
    mean = mean + new_accuracy
    accuracies.append(new_accuracy)

    if (no_of_queries != 0):
      # Processing committee members using k-means clustering
      from sklearn.cluster import KMeans
      kmeans = KMeans(n_clusters=6, init='k-means++',max_iter = 300, n_init = 10 , random_state = 0)
      y_means = kmeans.fit_predict(X_train)

      # Prepare two models on the basis of clusters
      cluster_1 = []
      cluster_2 = []
      cluster_3 = []
      cluster_4 = []
      cluster_5 = []
      cluster_6 = []
      for i in range(0,len(y_means)):
        if y_means[i] == 0:
          cluster_1.append(labeled_dataset[i,:])
        elif y_means[i] == 1:
          cluster_2.append(labeled_dataset[i,:])
        elif y_means[i] == 2:
          cluster_3.append(labeled_dataset[i,:])
        elif y_means[i] == 3:
          cluster_4.append(labeled_dataset[i,:])
        elif y_means[i] == 4:
          cluster_5.append(labeled_dataset[i,:])
        else:
          cluster_6.append(labeled_dataset[i,:])

      cluster_1 = np.array(cluster_1)
      cluster_2 = np.array(cluster_2)
      cluster_3 = np.array(cluster_3)
      cluster_4 = np.array(cluster_4)
      cluster_5 = np.array(cluster_5)
      cluster_6 = np.array(cluster_6)

      # Cluster 1 model M1
      cluster_1_X = cluster_1[:,:-1]
      cluster_1_y = cluster_1[:,-1]
      cluster_1_classifier = GaussianNB()
      cluster_1_classifier.fit(cluster_1_X,cluster_1_y)
      # Cluster 2 model M2
      cluster_2_X = cluster_2[:,:-1]
      cluster_2_y = cluster_2[:,-1]
      cluster_2_classifier = GaussianNB()
      cluster_2_classifier.fit(cluster_2_X,cluster_2_y)
      # Cluster 3 model M3
      cluster_3_X = cluster_3[:,:-1]
      cluster_3_y = cluster_3[:,-1]
      cluster_3_classifier = GaussianNB()
      cluster_3_classifier.fit(cluster_3_X,cluster_3_y)
      # Cluster 4 model M4
      cluster_4_X = cluster_4[:,:-1]
      cluster_4_y = cluster_4[:,-1]
      cluster_4_classifier = GaussianNB()
      cluster_4_classifier.fit(cluster_4_X,cluster_4_y)
      # Cluster 5 model M5
      cluster_5_X = cluster_5[:,:-1]
      cluster_5_y = cluster_5[:,-1]
      cluster_5_classifier = GaussianNB()
      cluster_5_classifier.fit(cluster_5_X,cluster_5_y)
      # Cluster 6 model M6
      cluster_6_X = cluster_6[:,:-1]
      cluster_6_y = cluster_6[:,-1]
      cluster_6_classifier = GaussianNB()
      cluster_6_classifier.fit(cluster_6_X,cluster_6_y)

print ("Final Average Accuracy : {}".format(mean/query_count))
print("Random Sampling Accuracy : {}".format(random_accuracy))
training_dataset = []
val = train_size + 1
for i in range(0,query_count):
  training_dataset.append(val)
  val = val + 1

plt.plot(training_dataset,accuracies)
plt.plot([min(training_dataset), max(training_dataset)], [random_accuracy]*2, color='red')
plt.title("QBC")
plt.xlabel("training dataset")
plt.ylabel("accuracies")
plt.show()
cm = confusion_matrix(y_test,y_pred)
print(cm)





