import argparse
from tests.test_topcoffea import test_datacard
 
def datacard_initial():
    try:
        test_datacard()
    except AssertionError as e:
        pass

def datacard_final():
    test_datacard()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run datacard maker tests')
    parser.add_argument('--final', action='store_true', help = 'Final run, allow assert to fail')
    args = parser.parse_args()
    final        = args.final
    
    if final:
        datacard_final()
    else:
        datacard_initial()
