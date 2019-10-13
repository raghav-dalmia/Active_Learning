import numpy as np
import pandas as pd
from statistics import mean
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


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


if __name__ == '__main__':
    dataset = pd.read_csv("./diabetes.csv").values[:, ]
    #dataset = pd.read_csv("./spambase.csv").values[:, ]
    #dataset = pd.read_csv("./bupa.csv").values[:, ]

    # imputing missing data
    imputer = SimpleImputer(missing_values=0, strategy="mean")
    imputer = imputer.fit(dataset[:, :-1])
    dataset[:, :-1] = imputer.transform(dataset[:, :-1])
    
    # feature scalling
    sc = StandardScaler()
    dataset[:, :-1] = sc.fit_transform(dataset[:, :-1])
    
    #clusters (representative data)
    clust = KMeans(n_clusters = ceil(dataset.shape[0]/10)).fit(dataset[:, :-1]).cluster_centers_
    
    itr = np.arange(0,11,1)
    
    ac1, ac2 = [],[]
    
    for it in itr:
        print(it)
        a1, a2 = [],[]
        for i in range(200):

            # shuffle dataset
            np.random.shuffle(dataset)

            # split data set into input and ouput set
            x = dataset[:, :-1]
            y = dataset[:, -1]

            #split data 1
            x_train, y_train, x_test, y_test, unlabel, label = split(x, y,0.3,0.6)
            
            #representative data
            ind = BallTree(unlabel).query(clust, return_distance=False, k=1).T[0]
            ind = np.unique(ind)
            
            x_rep = unlabel[ind, :]
            y_rep = label[ind]
            
            # svm parameters
            gama = 0.18
            c = 1

            for z in range(it):
                    
                #train model 1
                classifier1 = SVC(gamma = gama, C=c, probability=True)
                classifier1.fit(x_train, y_train)
                
                #uncertain points

		#support vectors 
                pts = classifier1.support_vectors_
		#points having probability between 0.47 to 0.53

                #p=0.47
                #probab = classifier1.predict_proba(x_train)[:,0]
                #pts_truth = np.array(list(map(lambda a: ((a>=p) and (a<=(1-p))) , probab)))
                #pts = x_train[pts_truth==True]                  
                
                try:
                    un_pt = BallTree(x_rep).query(pts, return_distance=False, dualtree=True).T[0]
                    un_pt = np.unique(un_pt)
                    
                    x_train = np.append(x_train, x_rep[un_pt, :], axis=0)
                    y_train = np.append(y_train, y_rep[un_pt])
                    
                    x_rep = np.delete(x_rep, un_pt, axis=0)
                    y_rep = np.delete(y_rep, un_pt)
                except:
                    pass
        
            #train model 1
            classifier1 = SVC(gamma = gama, C=c, probability=True)
            classifier1.fit(x_train, y_train)
            a1.append(classifier1.score(x_test, y_test))
                    
            #split data 2
            unlabel_size = 1 - ((x_train.shape[0]/x.shape[0]) + 0.299)
            x_train, y_train, x_test, y_test, unlabel, label = split(x, y,0.3,unlabel_size)
            
            #train model 2
            classifier2 = SVC(gamma = gama, C=c, probability=True)
            classifier2.fit(x_train, y_train)
            a2.append(classifier2.score(x_test, y_test))
                
        ac1.append(mean(a1)*100)
        ac2.append(mean(a2)*100)
        
    plt.plot(itr, ac1, color='orange', label='active learning')
    plt.plot(itr, ac2, color='blue', label='random sampling')
    plt.xlabel('number of iteration')
    plt.ylabel('accuracy')
    plt.title('query generation on the basis of supprot vector')
    plt.figtext(0.02, 0.04, "database : diabetes")
    plt.ylim(72, 76)
    plt.grid()
    plt.legend()
    plt.show()
