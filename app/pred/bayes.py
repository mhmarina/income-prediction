import argparse

datapath = "../data/adult.data.clean.csv"

def main():
    parser = argparse.ArgumentParser(description='Income Predictor')
    parser.add_argument('-a', '--age', required=False, help="continuous")
    parser.add_argument('-wc', '--workclass', required=False, help="[Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked]")
    parser.add_argument('-ed', '--education', required=False, help="[Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool]")
    parser.add_argument('-en', '--education-num', required=False, help="continuous")
    parser.add_argument('-o', '--occupation', required=False, help="[Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces]")
    parser.add_argument('-rel', '--relationship', required=False, help="[Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried]")
    parser.add_argument('-r', '--race', required=False, help="[White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black]")
    parser.add_argument('-s', '--sex', required=False, help="[Female, Male]")
    parser.add_argument('-cg', '--capital-gain', required=False, help="continuous")
    parser.add_argument('-cl', '--capital-loss', required=False, help="continuous")
    parser.add_argument('-hpw', '--hours-per-week', required=False, help="continuous")
    parser.add_argument('-nc', '--native-country', required=False, help="[United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands]")
    
if __name__ == '__main__':
    main()