#Linear Discriminant Analysis


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('bank-full.csv', delimiter = ';')

df.columns

#checking Missing data
pd.set_option('max_columns', None)
print(df.describe())
print(df.head(10))
df.info()
df.isnull().sum()
df.replace('unknown', np.nan, inplace= True)  #replace unknown with nan
df.info()
df.isnull().sum()
len(df),len(df.dropna())

# it seems from data features poutcome and contact are having large number of unknowns and also it seems to be insignificant
# so removing them
dataset= df.iloc[: , [0,1,2,3,4,5,6,7,9,10,11,12,13,14,16]]
dataset.info()
dataset.isnull().sum()
len(dataset),len(dataset.dropna())

# Missing Data Imputation
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer(strategy="most_frequent")
dataset.iloc[:,[1,3]] = my_imputer.fit_transform(dataset.iloc[:,[1,3]])
dataset.info()
dataset.isnull().sum()

dataset.describe()

#separating into input and output
X = dataset.iloc[:, 0:14]
y = dataset.iloc[:, [14]]

X.info()
X.columns

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X.iloc[:,[0,5,8,10,11,12,13]] = sc.fit_transform(X.iloc[:,[0,5,8,10,11,12,13]])

# creating dummy columns for the categorical columns 
dummies = pd.get_dummies(X[['job', 'marital', 'education', 'default','housing','loan','month']])
# Dropping the columns for which we have created dummies
X.drop(['job', 'marital', 'education', 'default','housing','loan','month'],inplace=True,axis = 1)

# adding the columns to the dataset data frame 
X = pd.concat([X,dummies],axis=1)

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA()
X = lda.fit_transform(X, y)
lda.explained_variance_ratio_
X = lda.transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
pd.crosstab(y_pred,y_test)
