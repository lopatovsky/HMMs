import numpy
cimport numpy
cimport cython

class HMM:

    def __init__(self):
        pass

    def meow(self):
        """Make the HMM to meow"""
        print('meow!')


def main():
    my_hmm = HMM()
    my_hmm.meow()

if __name__ == "__main__":
    main()
