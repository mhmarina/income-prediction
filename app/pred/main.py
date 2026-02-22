import argparse
from data import opt_wc, opt_s, opt_ed, opt_nc, opt_o, opt_r, opt_rel, opt_ms

def main():
    # python main.py --help
    parser = argparse.ArgumentParser(description='Income Predictor')
    parser.add_argument('-a', '--age', required=False, help="continuous")
    parser.add_argument('-wc', '--workclass', required=False, help=f"{opt_wc}")
    parser.add_argument('-ed', '--education', required=False, help=f"{opt_ed}")
    parser.add_argument('-en', '--education-num', required=False, help="continuous")
    parser.add_argument('-o', '--occupation', required=False, help=f"{opt_o}")
    parser.add_argument('-ms', '--marital-status', required=False, help=f"{opt_ms}")
    parser.add_argument('-rel', '--relationship', required=False, help=f"{opt_rel}")
    parser.add_argument('-r', '--race', required=False, help=f"{opt_r}")
    parser.add_argument('-s', '--sex', required=False, help=f"{opt_s}")
    parser.add_argument('-cg', '--capital-gain', required=False, help="continuous")
    parser.add_argument('-cl', '--capital-loss', required=False, help="continuous")
    parser.add_argument('-hpw', '--hours-per-week', required=False, help="continuous")
    parser.add_argument('-nc', '--native-country', required=False, help=f"{opt_nc}")
    
    args = parser.parse_args()

if __name__ == '__main__':
    main()