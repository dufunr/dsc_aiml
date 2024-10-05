import pandas 
import seaborn
import matplotlib.pyplot as mat
import numpy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x=r'/content/housing.csv'
y=pandas.read_csv(x)

#Task 1.1
print("first five rows of the dataset:")
print(y.head())

#Task 1.2
mat.figure(figsize=(12,8))
matrix = y.corr()
seaborn.heatmap(matrix,annot=True,linewidths=1)
mat.title('Correlation heatmap')
mat.show()

#Task 1.3
mat.figure(figsize=(10,5))
seaborn.scatterplot(x='RM', y='MEDV', data=y, color='red')
mat.title('Scatterplot of RM Vs MEDV')
mat.xlabel('RM')
mat.ylabel('MEDV')
mat.show()

#Task 2.1
if y.isnull().values.any() :
     y.dropna()
     print("missing values have been handled appropriately")
else:
     print("there are no missing values")

#Task 2.2
a=y.drop(columns=['MEDV'])
b=y['MEDV']
ss=StandardScaler()
s=ss.fit_transform(a)
sdf=pandas.DataFrame(s,columns=a.columns)
sdf['MEDV']=b
print("the normalized dataset is:")
print(sdf.head())

#Task 2.3
y['ratio'] = y['LSTAT'] / y['RM']
print("the dataset with the new column is:")
print(y.head())
y=y.drop(columns=['ratio'])

#Task 3.1
c=y.drop(columns=['MEDV'])
d=y['MEDV']
trainc, testc, traind, testd= train_test_split(c, d, test_size=0.20, shuffle=True)
lr=LinearRegression()
lr.fit(trainc,traind)
p=lr.predict(testc)
train = str(lr.score(trainc, traind) * 100)
test = str(lr.score(testc, testd) * 100)
print("linear regression train score:", train,"%")
print("linear regression test score:", test,"%")
mse=mean_squared_error(testd,p)
r=r2_score(testd,p)
print("the linear regression mean squared error is", mse)
print("the line regression R squared score is ", r)

#Task 3.2
rf = RandomForestRegressor(n_estimators=100)
rf.fit(trainc,traind)
p1=rf.predict(testc)
train1 = str(rf.score(trainc, traind) * 100)
test1 = str(rf.score(testc, testd) * 100)
print("forest regressor train score:", train1,"%")
print("forest regressor test score:", test1,"%")
mse1=mean_squared_error(testd,p1)
r1=r2_score(testd,p1)
print("the forest regressor mean squared error is", mse1)
print("the forest regressor R squared score is ", r1)
pred=pandas.DataFrame(testc, columns=c.columns)
pred['actual']=testd
pred['linear regression predicted']=p
pred['forest predicted']=p1
print("the dataset with predicted values are:")
print(pred)
imp = rf.feature_importances_
impdf = pandas.DataFrame({
    'Feature': trainc.columns,
    'Importance': imp})
impdf = impdf.sort_values(by='Importance', ascending=False)
mat.figure(figsize=(10, 5))
seaborn.barplot(x='Feature', y='Importance', data=impdf.head(5))
mat.xlabel('Features')
mat.ylabel('Importance')
mat.title('Top 5 important features')
mat.show()

#Task 4
new = numpy.array([[0.02, 19.1, 2.88, 0, 0.437, 6.999, 68.2, 4.2876, 1, 287.0, 11.3, 376.90, 4.18]])
newdf = pandas.DataFrame(new, columns=y.drop(columns=['MEDV']).columns) 
e=y.drop(columns=['MEDV'])
f=y['MEDV']
trainc, testc, traind, testd= train_test_split(e, f, test_size=0.20, shuffle=True)
testc = pandas.concat([testc, newdf], ignore_index=True) 
new_target = pandas.Series([0], index=[len(testd)])  
testd = pandas.concat([testd, new_target], ignore_index=True)
trc=ss.fit_transform(trainc)
tec=ss.transform(testc)
nn=Sequential()
nn.add(Dense(64, input_dim=trainc.shape[1], activation='relu'))
nn.add(Dense(1))
nn.compile(optimizer='adam', loss='mean_squared_error')
h =nn.fit(trc, traind, epochs=100, batch_size=32, validation_data=(tec, testd))
p2=nn.predict(tec)
p2 = p2.flatten()
testc['NN MEDV'] = p2
print("the dataset with predicted house prices are:")
print(testc)
print("the house price for a new data input has also been found and printed at the end")
mse2=mean_squared_error(testd,p2)
r2=r2_score(testd,p2)
print("the neural network mean squared error is", mse2)
print("the neural network R squared score is ", r2)
