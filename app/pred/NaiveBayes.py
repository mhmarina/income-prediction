from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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
        self.gnb = GaussianNB()
        self.gnb.fit(g_X, y)
        self.gnb_impute = SimpleImputer(strategy="mean")
        self.gnb_impute.fit(g_X)

        # # fit CategoricalNB to categorical data
        c_X = np.column_stack(list(cat.values())).astype(object)    
        self.cnb = Pipeline([
            ('encoder', OrdinalEncoder()), # CategoricalNB does not accept string classes natively
            ('clf', CategoricalNB())
        ])
        self.cnb.fit(c_X, y)
        self.cnb_impute = SimpleImputer(strategy="most_frequent")
        self.cnb_impute.fit(c_X)
    
    def predict(self, cont_features, cat_features):
        cat_features = self.cnb_impute.transform(cat_features)
        cont_features = self.gnb_impute.transform(cont_features)

        cat_prb = self.cnb.predict_log_proba(cat_features)
        con_prb = self.gnb.predict_log_proba(cont_features)
        final_prb = con_prb + cat_prb

        return opt_y[np.argmax(final_prb, axis=1)[0]]
def main():
    nc = NaiveBayesClassifier("../../data/adult.data.clean.csv")

    prd = nc.predict([[39, 13, 2174, 0, 40]], [['State-gov','Bachelors','Never-married','Adm-clerical', 'Not-in-family', 'White', 'Male', 'United-States']]) # <= 50
    print(f"output: {prd}")

    # pass in missing values
    prd_impute = nc.predict([[39, np.nan, 2174, np.nan, 40]], [['State-gov',np.nan,'Never-married','Adm-clerical', np.nan, 'White', 'Male', np.nan]]) # <= 50
    print(f"output with missing values: {prd_impute}")
    
if __name__ == '__main__':
    main()