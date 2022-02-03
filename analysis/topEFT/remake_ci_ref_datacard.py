import argparse
import tests.test_futures
 
def datacard():
    tests.test_futures.test_datacard_2l()
    tests.test_futures.test_datacard_2l_ht()
    tests.test_futures.test_datacard_3l()
    tests.test_futures.test_datacard_3l_ptbl()


if __name__ == '__main__':
    datacard()
