from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
import numpy as np

from data import data, target, continuous, categorical

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

        # # fit CategoricalNB to categorical data
        c_X = np.column_stack(list(cat.values()))
        self.cnb = Pipeline([
            ('encoder', OrdinalEncoder()), # CategoricalNB does not accept string classes natively
            ('clf', CategoricalNB())
        ])
        self.cnb.fit(c_X, y)

def main():
    nc = NaiveBayesClassifier("../../data/adult.data.clean.csv")
    print(nc.gnb.predict([[39, 13, 2174, 0, 40]])) # <= 50
    print(nc.cnb.predict([['State-gov','Bachelors','Never-married','Adm-clerical', 'Not-in-family', 'White', 'Male', 'United-States']])) # <= 50

if __name__ == '__main__':
    main()