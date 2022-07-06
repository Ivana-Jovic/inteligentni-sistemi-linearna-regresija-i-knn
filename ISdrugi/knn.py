import numpy as np
import pandas as pd
from scipy.stats.mstats_basic import mode


class KNN:
    listica = []
    listica2 = []
    def most_fr(self, li):
        return pd.Series(li).mode()[0]
        # .tolist()
        # val,cnt = mode(li,axis=0)
        # return val.ravel().tolist()

    def getNumOfNeigh(self, features):
        cnt= np.sqrt(len(features)).astype(int)
        if (cnt % 2) == 0 :
            cnt = cnt + 1
        return cnt
        # n je uk br- br komsija

    def predict(self, features,xtst,m,b0,b1,b2,b3,b4,b5):
        # a=pd.DataFrame(pr.to_numpy()[i])
        # features = features.copy(deep=True)
        n= len(features) - self.getNumOfNeigh(features)
        features = features.to_numpy()
        xtst = xtst.to_numpy()
        for j in range(len(features)):
        # features[["buying_price"]]-a[["buying_price"]

            # features[["dist"]]=np.sqrt(
            #         np.power(features[["buying_price"]]-b0,2)
            # + np.power(features[["maintenance"]]-b1,2)
            # +np.power(features[["doors"]]-b2,2)
            # +np.power(features[["seats"]]-b3,2)
            # +np.power(features[["trunk_size"]]-b4,2)
            # +np.power(features[["safety"]]-b5,2))

            # print(features[j][1])
            features[j][0] = np.sqrt(
            np.power(features[j][1] - b0, 2)
            + np.power(features[j][2] - b1, 2)
            + np.power(features[j][3]- b2, 2)
            + np.power(features[j][4]- b3, 2)
            + np.power(features[j][5] - b4, 2)
            + np.power(features[j][6]- b5, 2))

        # features=pd.DataFrame(features)
        # //sortiraj
        # features.sort_values(by=['dist'])
        features= features[np.argsort(features[:, 0])]
        # //uzmi prvih k
        # newfeatures=features.drop(features.tail(n).index, inplace= False)
        newFeatures= features[:-n]

        # vidi sta se najvise ponavlja
        # features[["pred"]]=newfeatures.mode()[0]
        # print(features[m][7])
        # features[m][7]=np.argmax(np.bincount(newFeatures[m][]))
        # xtst[m][7] = self.most_fr(newFeatures)[0]
        # self.listica.append(self.most_fr(newFeatures)[8])
        # print(newFeatures.T[8])
        self.listica.append(self.most_fr(newFeatures.T[8]))
        # print('====',self.most_fr(newFeatures)[0])
        # features = features.to_numpy()
        # return features.dot(self.coeff).reshape(-1, 1).flatten()
        return pd.DataFrame(xtst)

    def bla(self, X, X_test,y_test):
        m=-1

        for i in X_test.index:
            m=m+1
            # print(i)
            # a = pd.DataFrame(X_test.to_numpy()[i])
            X_test=self.predict(X,X_test,m, X_test.to_numpy()[m][1],X_test.to_numpy()[m][2],X_test.to_numpy()[m][3]
                         ,X_test.to_numpy()[m][4],X_test.to_numpy()[m][5],X_test.to_numpy()[m][6])
            self.listica2.append(float(y_test.to_numpy()[m]))
            # self.predict(X, a)
        return X_test
