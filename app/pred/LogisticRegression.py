from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from data import data, target, continuous, categorical
from pickle import dump, load
import os


class MyLogisticRegression():
    def __init__(self, datapath):
        self.datapath = datapath
        self.root_pickle = "pickle/LRC" #hehe
        self.pickle_model_path = f"{self.root_pickle}/model.pkl"
        self.pickle_cont_path = f"{self.root_pickle}/cont.pkl"
        self.pickle_cat_path = f"{self.root_pickle}/cat.pkl"
        self.pickle_stats_path = f"{self.root_pickle}/stats.pkl"

        self.model = ""
        self.cat_obj = {}
        self.cont_obj = {}
        self.stats = {}

        if os.path.exists(self.root_pickle):
            print("Loading model...")
            try:
                with open(self.pickle_model_path, "rb") as f:
                    self.model = load(f)
                with open(self.pickle_cat_path, "rb") as f:
                    self.cat_obj = load(f)
                with open(self.pickle_cont_path, "rb") as f:
                    self.cont_obj = load(f)
                with open(self.pickle_cont_path, "rb") as f:
                    self.cont_obj = load(f)
            except:
                self.train()
        else:
            print("Training model...")
            os.makedirs(self.root_pickle)
            self.train()
        self.train()
    
    def train(self):
        d = data(self.datapath)
        c_X = np.column_stack(list(categorical(d).values())).astype(object) 
        cat_impute = SimpleImputer(strategy="most_frequent")
        cat_impute.fit(c_X)
        enc = OneHotEncoder()
        enc.fit(c_X)
        c_X = enc.transform(c_X).toarray()
        self.cat_obj["impute"] = cat_impute
        self.cat_obj["enc"] = enc

        g_X = np.column_stack(list(continuous(d).values()))
        scaler = StandardScaler()
        scaler.fit(g_X)
        g_X = scaler.transform(g_X)
        cont_impute = SimpleImputer(strategy="mean")
        cont_impute.fit(g_X)
        self.cont_obj["scaler"] = scaler
        self.cont_obj["impute"] = cont_impute

        y = target(d)
        X = np.concatenate([c_X, g_X], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=42, test_size=0.2) 

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        # evaluate
        self.stats["accuracy"] = accuracy_score(y_test, y_pred)
        self.stats["precision"] = precision_score(y_test, y_pred, pos_label='>50K')
        self.stats["recall"] = recall_score(y_test, y_pred, pos_label='>50K')

        #pickle:
        with open(self.pickle_model_path, "wb") as f:
            dump(self.model, f, protocol=5)
        with open(self.pickle_cat_path, "wb") as f:
            dump(self.cat_obj, f, protocol=5)
        with open(self.pickle_cont_path, "wb") as f:
            dump(self.cont_obj, f, protocol=5)
        with open(self.pickle_stats_path, "wb") as f:
            dump(self.stats, f, protocol=5)        

    def predict(self, cont_features, cat_features):
        cat_features = self.cat_obj["impute"].transform(cat_features)
        cat_features = self.cat_obj["enc"].transform(cat_features).toarray()
        
        cont_features = self.cont_obj["scaler"].transform(cont_features)
        cont_features = self.cont_obj["impute"].transform(cont_features)
        features = np.concatenate([cat_features, cont_features], axis=1)
        return self.model.predict(features)

    def printMetrics(self):
        print("---Logistic Regression Model---")
        print(f"Accuracy: {self.stats["accuracy"]}")
        print(f"Precision: {self.stats["precision"]}")
        print(f"Recall: {self.stats["recall"]}")


def main():
    lr = MyLogisticRegression('../../data/adult.data.clean.csv')

    prd = lr.predict([[39, 13, 282023, 0, 56]], [['Private','Masters','Never-married','Adm-clerical', 'Not-in-family', 'White', 'Male', 'United-States']]) # <= 50
    print(f"output: {prd}") #expect >50k

    # pass in missing values
    prd_impute = lr.predict([[39, np.nan, 2174, np.nan, 40]], [['State-gov',np.nan,'Never-married','Adm-clerical', np.nan, 'White', 'Male', np.nan]]) # <= 50
    print(f"output with missing values: {prd_impute}") #expect <=50k

if __name__ == '__main__':
    main()