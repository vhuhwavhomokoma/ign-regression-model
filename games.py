import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Reading our DataFrame
Data = pd.read_csv('ign.csv')

#print data head
#print(Data.head())

#Drop unneeded variables
Data = Data.drop(["title","url","release_year","release_month","Unnamed: 0"], axis = 1)
#dummy categorical column
Data = pd.get_dummies(Data, drop_first=True)


x = Data.drop(columns=['score'])
y = Data['score']



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=10, shuffle=True)

model = Ridge().fit(X_train, y_train)

y_prediction = model.predict(X_train)
print("MAE on train data= " , mean_squared_error(y_train, y_prediction))
acc=r2_score(y_train,y_prediction)
print("Accuracy: "+ str(acc))

# Evaluating the trained model on test data
y_prediction = model.predict(X_test)
print("MAE on test data = " , mean_squared_error(y_test, y_prediction))
acc=r2_score(y_test,y_prediction)
print("Accuracy: "+ str(acc))

