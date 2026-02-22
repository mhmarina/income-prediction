from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from data import data, target, continuous, categorical, opt_y

class MyLogisticRegression():
    def __init__(self, datapath):
        self.datapath = datapath
        self.train()
    
    def train(self):
        d = data(self.datapath)
        c_X = np.column_stack(list(categorical(d).values())).astype(object) 

        self.cat_impute = SimpleImputer(strategy="most_frequent")
        self.enc = OneHotEncoder()
        self.cat_impute.fit(c_X)
        self.enc.fit(c_X)
        c_X = self.enc.transform(c_X).toarray()

        g_X = np.column_stack(list(continuous(d).values()))
        self.cont_impute = SimpleImputer(strategy="mean")
        self.cont_impute.fit(g_X)
        y = target(d)

        X = np.concatenate([c_X, g_X], axis=1)
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X = self.scaler.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=42, test_size=0.2) 

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        # evaluate
        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(y_test, y_pred, pos_label='>50K')
        self.recall = recall_score(y_test, y_pred, pos_label='>50K')

    def predict(self, cont_features, cat_features):
        cat_features = self.cat_impute.transform(cat_features)
        cont_features = self.cont_impute.transform(cont_features)
        cat_features = self.enc.transform(cat_features).toarray()

        features = np.concatenate([cat_features, cont_features], axis=1)
        self.scaler.transform(features)
        return self.model.predict(features)

def main():
    lr = MyLogisticRegression('../../data/adult.data.clean.csv')

    prd = lr.predict([[39, 13, 2174, 0, 40]], [['State-gov','Bachelors','Never-married','Adm-clerical', 'Not-in-family', 'White', 'Male', 'United-States']]) # <= 50
    print(f"output: {prd}")

    # pass in missing values
    prd_impute = lr.predict([[39, np.nan, 2174, np.nan, 40]], [['State-gov',np.nan,'Never-married','Adm-clerical', np.nan, 'White', 'Male', np.nan]]) # <= 50
    print(f"output with missing values: {prd_impute}")

if __name__ == '__main__':
    main()