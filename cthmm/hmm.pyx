import numpy
cimport numpy
cimport cython

class HMM:

    def __init__(self):
        """Initialize the HMM by small random values."""
        print("hello init")

    @classmethod
    def from_parameters( self,A,B,Pi):
        """Initialize the HMM by giving parameters - matrices A,B and vector Pi."""
        print("hello params")

    @classmethod
    def from_file( self,file_path ):
        """Initialize the HMM by the file from the file_path storing the parameters A,B,Pi""" ##TODO define the file format.
        print("hello file")

    def _init__(self,num):
        print(num)

    def meow(self):
        """Make the HMM to meow"""
        print('meow!')


def main():
    my_hmm = HMM()
    hmm2 = HMM.from_parameters(2,2,3)
    hmm2 = HMM.from_file("x.hmm")
    my_hmm.meow()

if __name__ == "__main__":
    main()
