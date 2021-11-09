import argparse
import tests.test_topcoffea
 
def datacard():
    tests.test_topcoffea.test_datacard_2l_nuis()
    tests.test_topcoffea.test_datacard_2l_ht()
    tests.test_topcoffea.test_datacard_3l()
    tests.test_topcoffea.test_datacard_3l_ptbl()


if __name__ == '__main__':
    datacard()
