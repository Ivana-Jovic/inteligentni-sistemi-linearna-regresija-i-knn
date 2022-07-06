import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from knn import KNN

# READ DATA
from knn import KNN

pd.set_option('display.max_columns', 9)
pd.set_option('display.width', None)
data = pd.read_csv('datasets/car_state.csv')

# DATA ANALYSIS
print("~ FIRST FIVE ROWS:")
print(data.head(), end='\n \n')
print("~ LAST FIVE ROWS:")
print(data.tail(), end='\n \n')
print("~ INFO:")
print(data.info(), end='\n \n')
print("~ STATISTICS:")
print(data.describe(), end='\n \n')
print("~ STATISTICS FOR STRINGS:")
print(data.describe(include=[object]), end='\n \n')


# DATA CLEANSING

# FEATURE ENGINEERING
data_train = data
labels = data['status']  # Series
le = LabelEncoder()
# data_train['buying_price'] = le.fit_transform(data_train['buying_price'],y=['1','2','3'])
# Pol leksikografski na 0 i 1
print('===',np.unique( data_train['safety']))
data_train['buying_price']= data_train['buying_price'].map({"low":1, "medium":2,"high":3,"very high":4})
data_train['maintenance']= data_train['maintenance'].map({"low":1, "medium":2,"high":3,"very high":4})
data_train['doors']= data_train['doors'].map({"2":1, "3":2,"4":3,"5 or more":4})
data_train['seats']= data_train['seats'].map({"2":1, "3":2,"4":3,"5 or more":4})
data_train['trunk_size']= data_train['trunk_size'].map({"big":1, "medium":2,"small":3})
data_train['safety']= data_train['safety'].map({"low":1, "medium":2,"high":3})
data_train['status']= data_train['status'].map({"acceptable":1, "good":2,"unacceptable":3,"very good":4})
print(data_train)

# -----------------------------------------------------
# # PLOTS
plot1= plt.figure(1)
plt.scatter(data_train['buying_price'], data_train['status'], c='blue')
plt.title('How status depends on buying_price')
plt.xlabel('buying_price')
plt.ylabel('status')
plt.savefig('test1.png', dpi=250)

plot2= plt.figure(2)
plt.scatter(data_train['maintenance'], data_train['status'], c='blue')
plt.title('How status depends on maintenance')
plt.xlabel('maintenance')
plt.ylabel('status')
plt.savefig('test2.png', dpi=250)

plot3= plt.figure(3)
plt.scatter(data_train['doors'], data_train['status'], c='blue')
plt.title('How status depends on doors')
plt.xlabel('doors')
plt.ylabel('status')
plt.savefig('test3.png', dpi=250)

plot4= plt.figure(4)
plt.scatter(data_train['seats'], data_train['status'], c='blue')
plt.title('How status depends on seats')
plt.xlabel('seats')
plt.ylabel('status')
plt.savefig('test4.png', dpi=250)

plot5= plt.figure(5)
plt.scatter(data_train['trunk_size'], data_train['status'], c='blue')
plt.title('How status depends on trunk_size')
plt.xlabel('trunk_size')
plt.ylabel('status')
plt.savefig('test5.png', dpi=250)

plot6= plt.figure(6)
plt.scatter(data_train['safety'], data_train['status'], c='blue')
plt.title('How status depends on safety')
plt.xlabel('safety')
plt.ylabel('status')
plt.savefig('test6.png', dpi=250)


#
# # -----------------------------------------------------
# # CORRELATIONS
# print('~ CORRELATIONS')
# print('buying_price correlation status ', data_train.status.corr(data_train.buying_price))
# print('maintenance correlation status ', data_train.status.corr(data_train.maintenance))
# print('doors correlation status ', data_train.status.corr(data_train.doors))
# print('seats correlation status ', data_train.status.corr(data_train.seats))
# print('trunk_size correlation status ', data_train.status.corr(data_train.trunk_size))
# print('safety correlation status ', data_train.status.corr(data_train.safety), end='\n \n')

# # data_train.drop("gender", axis=1, inplace=True)
# # data_train.drop("credit_card_debt", axis=1, inplace=True)



# -----------------------------------------------------
# # KNN
print('~ KNN - scikit')
# X_train = data_train[['buying_price', 'maintenance', 'doors', 'seats', 'trunk_size', 'safety']]
# y_train = data_train.status
X, X_test, y, y_test = train_test_split( data_train[['buying_price', 'maintenance', 'doors', 'seats', 'trunk_size', 'safety']],data_train.status, test_size= 0.3)

cnt= np.sqrt(len(data_train)).astype(int)
if (cnt % 2) == 0 :
    cnt = cnt + 1

KNN_model = KNeighborsClassifier(n_neighbors=cnt)
KNN_model.fit(X, y)

predictions = KNN_model.predict(X_test)
#
# print(pd.Series(predictions).map({1:"acceptable",2: "good", 3:"unacceptable",4:"very good"}), end='\n \n')
# print(pd.Series(y_test).map({1:"acceptable",2: "good", 3:"unacceptable",4:"very good"}), end='\n \n')

# print('rmse: ', mean_squared_error(np.array(y_test), predictions, squared=False))
# puta 100 = procenat uspesnosti predvidjanja
print('score: ', KNN_model.score(X_test, np.array(y_test)), end='\n \n')
print('classRep: ',end='\n')
print(classification_report(y_test,predictions,target_names=["acceptable","good","unacceptable","very good"]))
# ////////////////////
# --------------------
# # /////////////
listica3 = []
listica4 = []
for k in range(1,cnt):
    KNN_model = KNeighborsClassifier(n_neighbors=k)
    KNN_model.fit(X, y)
    listica3.append(KNN_model.score(X_test, np.array(y_test)))
    listica4.append(k)
plot9= plt.figure(9)
plt.plot(listica4,listica3, c='red')
plt.title('How score depends on numNeigh')
plt.xlabel('numNeigh')
plt.ylabel('score')
plt.savefig('test9.png', dpi=250)

# ---------------------------------
# # KNN ---
print('\n \n ----------------------------------------------')
print('~ KNN - notscikit')
X, X_test, y, y_test = train_test_split( data_train[['buying_price', 'maintenance', 'doors', 'seats', 'trunk_size', 'safety']],data_train.status, test_size= 0.3)



kn =KNN()
X.insert(0, 'dist', np.ones((len(X), 1)))
X.insert(7, 'pred', np.ones((len(X), 1)))
X_test.insert(0, 'dist', np.ones((len(X_test), 1)))
X_test.insert(7, 'pred', np.ones((len(X_test), 1)))


tmp= [X,y]
xyConcat=pd.concat(tmp,axis=1)
# print(bljuc)
kn.bla(xyConcat,X_test,y_test)
print(kn.listica)
print(kn.listica2)

print('score: ', metrics.accuracy_score(y_test,kn.listica), end='\n \n')
print('classRep: ',end='\n')
print(classification_report(y_test,kn.listica,target_names=["acceptable","good","unacceptable","very good"]))

# print(X_test.loc[X_test['pred']!= 1.0])
plt.show()