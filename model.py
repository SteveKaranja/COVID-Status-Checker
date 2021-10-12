import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle

warnings.simplefilter("ignore")

covid = pd.read_csv('corona.csv', low_memory=False)

# check gender column
covid['gender'].value_counts()

# drop data with gender as none.
covid.drop(covid.index[covid['gender'] == 'None'], inplace = True)

# check corona_result column
covid['corona_result'].value_counts()

# drop data with corona_result as other.
covid.drop(covid.index[covid['corona_result'] == 'other'], inplace = True)

# check cough column
covid['cough'].value_counts()

# As we can see, some records are filled in as 'None'.
# Let us assume that this meant that the patients had no cough symptoms, thus change the attribute from None to 0.

#  Change none to 0
covid['cough'] = covid['cough'].map({'0': 0, '1': 1, 'None': 0})

# check fever column
covid['fever'].value_counts()

#  Change none to 0
covid['fever'] = covid['fever'].map({'0': 0, '1': 1, 'None': 0})

# check sore_throat column
covid['sore_throat'].value_counts()

# check shortness_of_breath column
covid['shortness_of_breath'].value_counts()

# check headache column
covid['head_ache'].value_counts()

# check age_60_and_above column
covid['age_60_and_above'].value_counts()

# check test indication column
covid['test_indication'].value_counts()

#  Change other and abroad to 0, Contact with confirmed to 1
covid['test_indication'] = covid['test_indication'].map({'Other': 0, 'Abroad': 0, 'Contact with confirmed': 1})

# Processing data for model development

# Make gender column numerical
# male : 1 , female : 0
covid['gender'] = covid['gender'].map({'male': 1, 'female': 0})

# make corona_result column numerical
# positive : 1 , negative : 0
covid['corona_result'] = covid['corona_result'].map({'positive': 1, 'negative': 0})

# change columns that will be used for training the model to int datatype
covid = covid.astype({"cough": int, "fever": int, "sore_throat": int, "shortness_of_breath": int, "head_ache": int, "test_indication": int})

# Splitting the data into training and test datasets

x = covid.drop(labels = ["test_date", "corona_result", "age_60_and_above", "gender"], axis=1)
y = covid["corona_result"]

# Splitting the data into X train, X test and y train, y test

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

lg = LogisticRegression()
svc = SVC()

lg = lg.fit(X_train, y_train)
svc = svc.fit(X_train, y_train)

pickle.dump(lg, open('lg.pkl', 'wb'))
pickle.dump(svc, open('svc.pkl', 'wb'))