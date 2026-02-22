import argparse
from LogisticRegression import MyLogisticRegression
from NaiveBayes import NaiveBayesClassifier
import numpy as np
from data import opt_wc, opt_s, opt_ed, opt_nc, opt_o, opt_r, opt_rel, opt_ms, continuous, categorical, data


def main():
    datapath = '../../data/adult.data.clean.csv'
    d = data(datapath)
    features = list(continuous(d).keys()) + list(categorical(d).keys())

    # python main.py --help
    parser = argparse.ArgumentParser(description='Income Predictor')
    parser.add_argument('-model', '--model', required=True, help="Naive Bayes, Logistic Regression", choices=['Naive Bayes', 'NB', 'Logistic Regression', 'LR'])

    parser.add_argument('-a', '--age', required=False, help="continuous", type=int)
    parser.add_argument('-en', '--education-num', required=False, help="continuous", type=int)
    parser.add_argument('-cg', '--capital-gain', required=False, help="continuous", type=int)
    parser.add_argument('-cl', '--capital-loss', required=False, help="continuous", type=int)
    parser.add_argument('-hpw', '--hours-per-week', required=False, help="continuous", type=int)

    parser.add_argument('-wc', '--workclass', required=False, help=f"{opt_wc}", choices=opt_wc)
    parser.add_argument('-ed', '--education', required=False, help=f"{opt_ed}", choices=opt_ed)
    parser.add_argument('-o', '--occupation', required=False, help=f"{opt_o}", choices=opt_o)
    parser.add_argument('-ms', '--marital-status', required=False, help=f"{opt_ms}", choices=opt_ms)
    parser.add_argument('-rel', '--relationship', required=False, help=f"{opt_rel}", choices=opt_rel)
    parser.add_argument('-r', '--race', required=False, help=f"{opt_r}", choices=opt_r)
    parser.add_argument('-s', '--sex', required=False, help=f"{opt_s}", choices=opt_s)
    parser.add_argument('-nc', '--native-country', required=False, help=f"{opt_nc}", choices=opt_nc)
    
    args = parser.parse_args()

    cont = [args.age, args.education_num, args.capital_gain, args.capital_loss, args.hours_per_week]
    cat = [
        args.workclass,
        args.education,
        args.marital_status,
        args.occupation,
        args.relationship,
        args.race,
        args.sex,
        args.native_country
    ]

    for i in range(len(cont)):
        if cont[i] == None:
            cont[i] = np.nan
    
    for i in range(len(cat)):
        if cat[i] == None:
            cat[i] = np.nan

    model_type = args.model

    if model_type in ['Naive Bayes', 'NB']:
        model = NaiveBayesClassifier(datapath)
    elif model_type in [ 'Logistic Regression', 'LR']:
        model = MyLogisticRegression(datapath)

        # > 100 coefficients; one hot encoding expands the number of features
        # ex: 'x0_Federal-gov' 'x0_Local-gov' 'x0_Private' 'x0_Self-emp-inc'
        #  'x0_Self-emp-not-inc' 'x0_State-gov' 'x0_Without-pay' 'x1_10th' 'x1_11th'
        # print(model.model.coef_)
        # unwrap coefficients
        coeff = [0,0,0,0,0,[],[],[],[],[],[],[],[]]
        cat_names = model.enc.get_feature_names_out()
        
        # we can just use the first 5 continuous coefficients
        for i in range(5):
            coeff[i] = model.model.coef_[0][i]
        for i in range(len(cat_names)):
            ft_index = int(cat_names[i].split('_')[0][1])
            val = model.model.coef_[0][i+5]
            coeff[5+ft_index].append(val)
        for i in range(5, len(coeff)):
            coeff[i] = sum(coeff[i])/len(coeff[i])
        
        # for i in range(13):
        #     # positive class is >50
        #     print(f"{features[i]}: {coeff[i]}")

    print(model.predict([cont], [cat])[0])
    print(model.getMetrics())

if __name__ == '__main__':
    main()