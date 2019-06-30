import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from random import shuffle
from sklearn.svm import SVC
from math import sqrt, ceil


def split(x, y, test_size=0.25, validation_size=0.7):
    p = x[y == 1]
    n = x[y == 0]
    test_size_p = int(test_size * p.shape[0])
    test_size_n = int(test_size * n.shape[0])
    validation_size_p = int(validation_size * p.shape[0])
    validation_size_n = int(validation_size * n.shape[0])
    x_test = np.append(p[:test_size_p, :], n[:test_size_n, :], axis=0)
    y_test = np.append(np.ones(test_size_p), np.zeros(test_size_n))
    unlabel = np.append(p[test_size_p:test_size_p+validation_size_p, :],
                        n[test_size_n:test_size_n+validation_size_n, :], axis=0)
    labels = np.append(np.ones(validation_size_p), np.zeros(validation_size_n))
    x_train = np.append(p[test_size_p+validation_size_p:, :],
                        n[test_size_n+validation_size_n:, :], axis=0)
    y_train = np.append(np.ones(p.shape[0] - test_size_p - validation_size_p),
                        np.zeros(n.shape[0] - test_size_n - validation_size_n))
    return x_train, y_train, x_test, y_test, unlabel, labels


def mid_point(p1, p2):
    p1 = p1[np.newaxis, :]
    p2 = p2[np.newaxis, :]
    return np.mean(np.append(p1, p2, axis=0), axis=0)


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))


def query(pos, unlable, labels):
    ans = labels[pos]
    #print("Label for " + str(unlabel[pos]) + " : " + str(ans))
    labels = np.delete(labels, pos)
    unlable = np.delete(unlable, pos, 0)
    return ans, unlable, labels


if __name__ == '__main__':
    dataset = pd.read_csv("./data.csv").values[:, ]
    #dataset = pd.read_csv("./data.csv").values[:, [1, 2, -1]]
    #dataset = pd.read_csv("./sample_data.csv").values[:, ]

    # imputing missing data
    imputer = SimpleImputer(missing_values=0, strategy="mean")
    imputer = imputer.fit(dataset[:, :-1])
    dataset[:, :-1] = imputer.transform(dataset[:, :-1])

    # shuffle dataset
    np.random.shuffle(dataset)

    # split data set into input and ouput set
    x = dataset[:, :-1]
    y = dataset[:, -1]

    # feature scalling
    sc = StandardScaler()
    x = sc.fit_transform(x)

    # split data set
    # label = 5% test = 25% unlabel = 70%
    x_train, y_train, x_test, y_test, unlabel, labels = split(x, y)

    num_clusters = ceil(x.shape[0]/10)  # 77
    kmean = KMeans(n_clusters=num_clusters, random_state=0, n_init=20).fit(x)
    unlabel_pool_center = kmean.cluster_centers_  # [77,1]

    ind = BallTree(unlabel, leaf_size=5).query(
        unlabel_pool_center, k=1, return_distance=False).T  # [77,1]
    representative_data = unlabel[ind, :][0]  # [77,8]

    # svm parameters
    gama = 0.18
    c = 0.961

    # fit model
    classifier1 = SVC(gamma=gama, C=c)
    classifier1.fit(x_train, y_train)
    print(classifier1.score(x_test, y_test))

    #predict output for x_train
    y_pred = classifier1.predict(x_train)

    #p contains all datapoints lie in positive part
    #n contains all datapoints lie in negative part
    p = x_train[y_pred == 1]
    n = x_train[y_pred == 0]

    #centroid of p and n
    c_p = np.mean(p, axis=0)
    c_n = np.mean(n, axis=0)

    #start ploting window
    fig, (ax1, ax2) = plt.subplots(1,2)

    
    # to find closest and opposite pair near boundary
    while(True):

        # mid point of centroid
        x_s = mid_point(c_p, c_n)

        # nearest neighbour of mid point
        i_q = BallTree(representative_data, leaf_size=5).query(x_s[np.newaxis,:], k=1, return_distance=False)
        x_q = representative_data[i_q][0]

        #query it's label
        query_ans, representative_data, labels = query(i_q, representative_data, labels)

        #check if the queried point is valid or not
        dist = euclidean_distance(c_p, c_n)
        pred_pos = classifier1.predict(x_q)
        if(pred_pos == 1):
            if dist > euclidean_distance(x_q, c_n):
                c_p = x_q[0]
            else:
                break
        else:
            if dist > euclidean_distance(c_p, x_q):
                c_n = x_q[0]
            else:
                break
        
        #if point is valid append it in training set
        x_train = np.append(x_train, x_q, axis=0)
        y_train = np.append(y_train, query_ans)
        ax1.scatter(x_q[0][0], x_q[0][1],s=12, color='grey')


            
    #fit second model
    classifier2 = SVC(gamma=gama, C=c)
    classifier2.fit(x_train,y_train)
    print(classifier2.score(x_test,y_test))

    #split points according to their original label
    p = x_train[y_train == 1]
    n = x_train[y_train == 0]
    
    '''
    #plot1
    ax1.scatter(c_n[0], c_n[1], color='orange')
    ax1.scatter(c_p[0], c_p[1], color='orange')
    x1, x2 = np.meshgrid(np.arange(x[:, 0].min(
    )-1, x[:, 0].max()+1, 0.01), np.arange(x[:, 1].min()-1, x[:, 1].max()+1, 0.01))
    #vec = classifier1.support_vectors_
    #ax1.scatter(vec[:,0], vec[:,1], s=10, color='orange')
    ax1.scatter(p[:, 0], p[:, 1], color='red', s=2,label = "postive")
    ax1.scatter(n[:, 0], n[:, 1], color='green', s=2, label = "negative")
    ax1.contourf(x1, x2, classifier1.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(
        x1.shape), alpha=0.2, cmap=ListedColormap(('green', 'red')))
    plt.figtext(0.02,0.04 ,"Accuracy : " + str(classifier1.score(x_test, y_test)))

    #plot2
    x1, x2 = np.meshgrid(np.arange(x[:, 0].min(
    )-1, x[:, 0].max()+1, 0.01), np.arange(x[:, 1].min()-1, x[:, 1].max()+1, 0.01))
    #vec = classifier1.support_vectors_
    #ax2.scatter(vec[:,0], vec[:,1], s=10, color='orange')
    ax2.scatter(p[:, 0], p[:, 1], color='red', s=2,label = "postive")
    ax2.scatter(n[:, 0], n[:, 1], color='green', s=2, label = "negative")
    ax2.contourf(x1, x2, classifier2.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(
        x1.shape), alpha=0.2, cmap=ListedColormap(('green', 'red')))
    plt.figtext(0.02,0.02 ,"New Accuracy : " + str(classifier2.score(x_test, y_test)))


    plt.legend()
    plt.show()
    '''