import numpy as np

opt_wc = "Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked".split(", ")
opt_ed = "Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool".split(", ")
opt_o = "Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces".split(", ")
opt_ms = "Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse".split(", ")
opt_rel = "Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried".split(", ")
opt_r = "White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black".split(", ")
opt_s = "Female, Male".split(", ")
opt_nc = "United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands".split(", ")
opt_y = ["<=50K", ">50K"]

def data(datapath):
    return np.genfromtxt(
        datapath,     
        delimiter=",",
        dtype=None,         
        encoding="utf-8",
        autostrip=True     
    )

def target(d):
    return d['f14']

def continuous(d): 
    return {
        'age': d['f0'],
        'education-num': d['f4'],
        'capital-gain': d['f10'],
        'capital-loss': d['f11'],
        'hours-per-week': d['f12']
    }

def categorical(d): 
    return {
        'workclass': d['f1'],
        'education': d['f3'],
        'marital-status': d['f5'],
        'occupation': d['f6'],
        'relationship': d['f7'],
        'race': d['f8'],
        'sex': d['f9'],
        'native-country': d['f13']
    }