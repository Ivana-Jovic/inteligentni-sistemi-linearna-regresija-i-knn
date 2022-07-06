import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from LinearRegressionGradientDescent import LinearRegressionGradientDescent
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# READ DATA
pd.set_option('display.max_columns', 7)
pd.set_option('display.width', None)
data = pd.read_csv('datasets/car_purchase.csv')
data.drop("customer_id", axis=1, inplace=True)

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
# ..Nema sta, vec smo sklonili rb
# FEATURE ENGINEERING
data_train = data
labels = data[ 'max_purchase_amount']  # Series
le = LabelEncoder()
data_train['gender'] = le.fit_transform(data_train['gender'])  # Pol leksikografski na 0 i 1

# -----------------------------------------------------
# PLOTS
plot1= plt.figure(1)
plt.scatter(data_train['age'], data_train['max_purchase_amount'], c='blue')
plt.title('How max purchase amount depends on age')
plt.xlabel('age')
plt.ylabel('max purchase amount')
plt.savefig('test1.png', dpi=250)

plot2= plt.figure(2)
plt.scatter(data_train['annual_salary'], data_train['max_purchase_amount'], c='green')
plt.title('How max purchase amount depends on annual salary')
plt.xlabel('annual salary in $')
plt.ylabel('max purchase amount')
plt.savefig('test2.png', dpi=250)

plot3= plt.figure(3)
plt.scatter(data_train['credit_card_debt'], data_train['max_purchase_amount'], c='red')
plt.title('How max purchase amount depends on credit card debt')
plt.xlabel('credit_card_debt')
plt.ylabel('max purchase amount')
plt.savefig('test3.png', dpi=250)

plot4= plt.figure(4)
plt.scatter(data_train['net_worth'], data_train['max_purchase_amount'], c='purple')
plt.title('How max purchase amount depends on net worth')
plt.xlabel('net_worth')
plt.ylabel('max purchase amount')
plt.savefig('test4.png', dpi=250)

# plot5= plt.figure(5)
# plt.scatter(data_train['gender'], data_train['max_purchase_amount'], c='orange')
# plt.title('How max purchase amount depends on gender')
# plt.xlabel('gender')
# plt.ylabel('max purchase amount')
# plt.savefig('test5.png', dpi=250)


# -----------------------------------------------------
# CORRELATIONS
print('~ CORRELATIONS')
print('Gender correlation max_purchase_amount ', data_train.max_purchase_amount.corr(data_train.gender))
print('Age correlation max_purchase_amount ', data_train.max_purchase_amount.corr(data_train.age))
print('Annual salary correlation max_purchase_amount ', data_train.max_purchase_amount.corr(data_train.annual_salary))
print('CCDebt correlation max_purchase_amount ', data_train.max_purchase_amount.corr(data_train.credit_card_debt))
print('Net worth correlation max_purchase_amount ', data_train.max_purchase_amount.corr(data_train.net_worth), end='\n \n')

# data_train.drop("gender", axis=1, inplace=True)
# data_train.drop("credit_card_debt", axis=1, inplace=True)

# -----------------------------------------------------
# LINEAR REGRESSION
print('~ LINEAR REGRESSION - scikit')
# X_train = data_train[['age', 'annual_salary', 'net_worth']]
# y_train = data_train.max_purchase_amount
X_train, X_test, y_train, y_test = train_test_split(data_train[['age', 'annual_salary', 'net_worth']],data_train.max_purchase_amount, test_size= 0.2)
LR_model = LinearRegression()
LR_model.fit(X_train, y_train)

# predictions = LR_model.predict(data_train[['age', 'annual_salary', 'net_worth']])
predictions = LR_model.predict(X_test)

print('h =', end=" ")
print(LR_model.coef_[0], '* age  + ',
LR_model.coef_[1], '* annual_salary +',
LR_model.coef_[2], '* net_worth +',
LR_model.intercept_)


print('rmse: ', mean_squared_error(np.array(y_test), predictions, squared=False))
# print('mse: ', mean_squared_error(np.array(y_train), predictions, squared=True))
# puta 100 = procenat uspesnosti predvidjanja
print('score: ', LR_model.score(X_test, np.array(y_test)), end='\n \n')
# -----------------------------------------------------
# LINEAR REGRESSION not scikit
print('~ LINEAR REGRESSION - not scikit')

LRgd_model = LinearRegressionGradientDescent()
X = data_train[['age', 'annual_salary', 'net_worth']].copy(deep=True)
y = data_train.max_purchase_amount.copy(deep=True)

X, X_test, y, y_test = train_test_split(X,y, test_size= 0.3)

XX = X.copy(deep=True)
X[['age']] = X[['age']] / X['age'].max()
X[['annual_salary']] = X[['annual_salary']] / X['annual_salary'].max()
X[['net_worth']] = X[['net_worth']] / X['net_worth'].max()
yyy=y.copy(deep=True)
y=y/y.max()

X_test[['age']] = X_test[['age']] / XX[['age']].max()
X_test[['annual_salary']] = X_test[['annual_salary']] / XX[['annual_salary']].max()
X_test[['net_worth']] = X_test[['net_worth']] / XX[['net_worth']].max()
# y_test=y_test/y_test.max()

LRgd_model.fit(X, y)
learning_rates = np.array([[0.9], [0.9], [0.9], [0.9]])
res_coeff, mse_history = LRgd_model.perform_gradient_descent(learning_rates, 1000)
# kao fja cost iz klase
predicted = LRgd_model.predict(X_test) * yyy.max()


print('h =', end=" ")
print(res_coeff[1]*yyy.max()/ XX['age'].max(), '* age  +',
res_coeff[2]*yyy.max() / XX['annual_salary'].max(), '* annual_salary +',
res_coeff[3]*yyy.max()/ XX['net_worth'].max(), '* net_worth +',
res_coeff[0]*yyy.max(), end="\n")

print('rmse: ', mean_squared_error(np.array(y_test), predicted, squared=False))
# puta 100 = procenat uspesnosti predvidjanja
print('score: ', r2_score(np.array(y_test), predicted), end='\n \n')


print('----predicted', end="\n")
print(pd.DataFrame(predicted).head(), end="\n \n")
print('----notPredicted', end="\n")
print(y_test.head())

# ////////////////////
plot6= plt.figure(6)
plt.plot(np.arange(0, len(mse_history), 1), np.array(mse_history)*yyy.max())
plt.title('How mse depends on numIt')
plt.xlabel('numIter')
plt.ylabel('mse')
plt.savefig('test6.png', dpi=250)

plt.show()
