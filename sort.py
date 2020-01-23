from mpi4py import MPI
import numpy as np
from sys import argv
COMM = MPI.COMM_WORLD


def echange_min(p, me, tab):
    if p > me >= 0:
        data = COMM.recv(source=me)
        tab2 = data["tab2"]
        data = {"tab2": tab}
        COMM.send(data, dest=me)
        z = np.concatenate((tab, tab2))
        z.sort()
        n = tab.size
        return z[:n]
    return tab


def echange_max(p, me, tab):
    if p > me >= 0:
        data = {"tab2": tab}
        COMM.send(data, dest=me)
        data = COMM.recv(source=me)
        tab2 = data["tab2"]
        z = np.concatenate((tab, tab2))
        z.sort()
        n = tab.size
        return z[z.size - n:]
    return tab


def sort():
    p = COMM.Get_size()
    me = COMM.Get_rank()
    if me == 0:
        numbers = np.genfromtxt("array.csv", delimiter=",")
        tabs = np.array_split(numbers, p)
        print("tableau non trie ==> ", numbers)
    else:
        tabs = None
    tab = COMM.scatter(tabs, root=0)
    tab.sort()
    for e in range(p):
        if e % 2 == 0:
            if me % 2 == 0:
                tab = echange_min(p, me + 1, tab)
            else:
                tab = echange_max(p, me - 1, tab)
        else:
            if me % 2 == 0:
                tab = echange_max(p, me - 1, tab)
            else:
                tab = echange_min(p, me + 1, tab)
    array = COMM.gather(tab, root=0)
    if me == 0:
        tableau = []
        for i in range(len(array)):
            tableau.extend(array[i].tolist())
        print("tableau apres tri ==> ", np.array(tableau))


if __name__ == '__main__':
    sort()