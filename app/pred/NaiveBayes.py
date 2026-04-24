from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np
from data import data, target, continuous, categorical, opt_y
from pickle import dump, load
import os

class NaiveBayesClassifier:
    def __init__(self, datapath):
        self.datapath = datapath
        self.root_pickle = "pickle/NBC" #hehe
        self.pickle_cnb_path = f"{self.root_pickle}/cnb.pkl"
        self.pickle_gnb_path = f"{self.root_pickle}/gnb.pkl"
        self.pickle_stats_path = f"{self.root_pickle}/stats.pkl"

        self.stats = {}
        self.cnb_obj = {}
        self.gnb_obj = {}

        if os.path.exists(self.root_pickle):
            print("Loading model...")
            with open(self.pickle_cnb_path, "rb") as f:
                self.cnb_obj = load(f)
            with open(self.pickle_gnb_path, "rb") as f:
                self.gnb_obj = load(f)
            with open(self.pickle_stats_path, "rb") as f:
                self.stats = load(f)
        else:
            print("Training model...")
            os.makedirs(self.root_pickle)
            self.train()

    def train(self):
        d = data(self.datapath)
        cat = categorical(d)
        cont = continuous(d)
        y = target(d)

        g_X = np.column_stack(list(cont.values()))
        c_X = np.column_stack(list(cat.values())).astype(object)  

        X_train, X_test, y_train, y_test = train_test_split(
            np.arange(len(y)), y, stratify=y, random_state=42, test_size=0.2
        )
        gX_train = g_X[X_train]
        gX_test = g_X[X_test]
        cX_train = c_X[X_train]
        cX_test = c_X[X_test]

        # fit GaussianNB model to continuous data
        gnb = GaussianNB()
        gnb_impute = SimpleImputer(strategy="mean")
        gnb_impute.fit(gX_train)
        gnb.fit(gX_train, y_train)

        # fit CategoricalNB to categorical data
        cnb = Pipeline([
            ('encoder', OrdinalEncoder()), # CategoricalNB does not accept string classes natively
            ('clf', CategoricalNB())
        ])
        cnb_impute = SimpleImputer(strategy="most_frequent")
        cnb_impute.fit(cX_train)
        cnb.fit(cX_train, y_train)

        self.cnb_obj["model"] = cnb
        self.cnb_obj["impute"] = cnb_impute

        self.gnb_obj["model"] = gnb
        self.gnb_obj["impute"] = gnb_impute

        # accuracy
        Y_pred = self.predict(gX_test, cX_test)
        self.stats["accuracy"] = accuracy_score(y_test, Y_pred)
        self.stats["precision"] = precision_score(y_test, Y_pred, pos_label='>50K')
        self.stats["recall"] = recall_score(y_test, Y_pred, pos_label='>50K')
        
        # pickle:
        with open(self.pickle_cnb_path, "wb") as f:
            dump(self.cnb_obj, f, protocol=5)
        with open(self.pickle_gnb_path, "wb") as f:
            dump(self.gnb_obj, f, protocol=5)
        with open(self.pickle_stats_path, "wb") as f:
            dump(self.stats, f, protocol=5)
            
    def predict(self, cont_features, cat_features):
        # enc = self.cnb.named_steps['encoder']
        # for i, cats in enumerate(enc.categories_):
        #     print(f"Feature {i}: {cats}")
        # print(cat_features)
        cat_features = self.cnb_obj["impute"].transform(cat_features)
        cont_features = self.gnb_obj["impute"].transform(cont_features)

        cat_prb = self.cnb_obj["model"].predict_log_proba(cat_features)
        con_prb = self.gnb_obj["model"].predict_log_proba(cont_features)
        final_prb = con_prb + cat_prb

        return np.array(opt_y)[np.argmax(final_prb, axis=1)]
    
    def printMetrics(self):
        print("---Naive Bayes Classifier---")
        print(f"Accuracy: {self.stats["accuracy"]}")
        print(f"Precision: {self.stats["precision"]}")
        print(f"Recall: {self.stats["recall"]}")
    
def main():
    nc = NaiveBayesClassifier("../../data/adult.data.clean.csv")

    prd = nc.predict([[39, 13, 282023, 0, 56]], [['Private','Masters','Never-married','Adm-clerical', 'Not-in-family', 'White', 'Male', 'United-States']]) # <= 50
    print(f"output: {prd}") #expect >50k

    # pass in missing values
    prd_impute = nc.predict([[39, np.nan, 2174, np.nan, 40]], [['State-gov',np.nan,'Never-married','Adm-clerical', np.nan, 'White', 'Male', np.nan]]) # <= 50
    print(f"output with missing values: {prd_impute}") #expect <=50k

if __name__ == '__main__':
    main()