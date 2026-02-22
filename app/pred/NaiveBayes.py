from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np
from data import data, target, continuous, categorical, opt_y

class NaiveBayesClassifier:
    def __init__(self, datapath):
        self.datapath = datapath
        self.train()

    def train(self):
        d = data(self.datapath)
        cat = categorical(d)
        cont = continuous(d)
        y = target(d)

        # fit GaussianNB model to continuous data
        g_X = np.column_stack(list(cont.values()))
        gX_train, gX_test, Y_train, Y_test = train_test_split(g_X, y, random_state=42, test_size=0.2)
        self.gnb = GaussianNB()
        self.gnb.fit(gX_train, Y_train)
        self.gnb_impute = SimpleImputer(strategy="mean")
        self.gnb_impute.fit(gX_train)

        # fit CategoricalNB to categorical data
        c_X = np.column_stack(list(cat.values())).astype(object)  
        cX_train, cX_test, Y_train, Y_test = train_test_split(
            c_X,
            y,
            stratify=y, # ensures each class is represented in test and train sets
            random_state=42, test_size=0.2)  
        
        self.cnb = Pipeline([
            ('encoder', OrdinalEncoder()), # CategoricalNB does not accept string classes natively
            ('clf', CategoricalNB())
        ])
        self.cnb.fit(cX_train, Y_train)
        self.cnb_impute = SimpleImputer(strategy="most_frequent")
        self.cnb_impute.fit(cX_train)

        # accuracy
        Y_pred = self.predict(gX_test, cX_test)
        self.accuracy = accuracy_score(Y_test, Y_pred)
        self.precision = precision_score(Y_test, Y_pred, pos_label='>50K')
        self.recall = recall_score(Y_test, Y_pred, pos_label='>50K')
            
    def predict(self, cont_features, cat_features):
        cat_features = self.cnb_impute.transform(cat_features)
        cont_features = self.gnb_impute.transform(cont_features)

        cat_prb = self.cnb.predict_log_proba(cat_features)
        con_prb = self.gnb.predict_log_proba(cont_features)
        final_prb = con_prb + cat_prb
        return np.array(opt_y)[np.argmax(final_prb, axis=1)]
    
    def getMetrics(self):
        print("---Naive Bayes Classifier---")
        print(f"Accuracy: {self.accuracy}")
        print(f"Precision: {self.precision}")
        print(f"Recall: {self.recall}")
    
def main():
    nc = NaiveBayesClassifier("../../data/adult.data.clean.csv")

    prd = nc.predict([[39, 13, 2174, 0, 40]], [['State-gov','Bachelors','Never-married','Adm-clerical', 'Not-in-family', 'White', 'Male', 'United-States']]) # <= 50
    print(f"output: {prd}")

    # pass in missing values
    prd_impute = nc.predict([[39, np.nan, 2174, np.nan, 40]], [['State-gov',np.nan,'Never-married','Adm-clerical', np.nan, 'White', 'Male', np.nan]]) # <= 50
    print(f"output with missing values: {prd_impute}")

if __name__ == '__main__':
    main()