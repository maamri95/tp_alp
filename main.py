import os
import numpy as np

if __name__ == '__main__':
    n = input("entre le nombre de processus s'il vous plait :")
    os.system("mpiexec -n {} python -m mpi4py sort.py".format(n))