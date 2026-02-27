The purpose of this project is to predict income for individuals using ML models and visualize the given dataset.
```.
├── app
│   ├── pred 
│   │   ├── LogisticRegression.py # Logistic Regression Model using ScikitLearn
│   │   ├── NaiveBayes.py         # Naive Bayes Prediction Model: Gaussian for continuous variables, Categorical otherwise (ScikitLearn). Total Probability is a product of both
│   │   ├── data.py               
│   │   └── main.py               # Provides a command line interface for interacting and inference with either model
│   └── viz.ipynb                # Various visualizations of dataset and model weights/probabilities using matplotlib
├── data
│   ├── adult.data.clean.csv     # cleaned up dataset for use
│   ├── adult.data.csv           # census dataset
│   └── adult.names.txt          # description of dataset
└── scripts
    └── clean.py                 # dataset cleanup script
```
