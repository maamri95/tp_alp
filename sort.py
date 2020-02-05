import numpy as np
from mpi4py import MPI

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()  # numbre de processus
RANK = COMM.Get_rank()  # le processus actif
send = COMM.send  # fonction d'envoi
recv = COMM.recv  # fonction de reception


def echange_min(me, tab):
    """
        recupere le n min des valeurs entre le la partie de process RANK et le process me,
        n etant la taille de la partie de RANK
        :param me: process adjacent a RANK
        :param tab: partie de process RANK
        :return: la nouvelle partie de process RANK
    """
    if not (SIZE > me >= 0):
        return tab
    data = recv(source=me)
    tab2 = data["tab2"]
    data = {"tab2": tab}
    send(data, dest=me)
    z = np.concatenate((tab, tab2))
    z.sort()
    n = tab.size
    return z[:n]


def echange_max(me, tab):
    """
    recupere le n max des valeurs entre le la partie de process RANK et le process me,
    n etant la taille de la partie de RANK
    :param me: process adjacent a RANK
    :param tab: partie de process RANK
    :return: la nouvelle partie de process RANK
    """
    if not (SIZE > me >= 0):
        return tab
    data = {"tab2": tab}
    send(data, dest=me)
    data = recv(source=me)
    tab2 = data["tab2"]
    z = np.concatenate((tab, tab2))
    z.sort()
    n = tab.size
    return z[z.size - n:]


def sort():
    """
    tri a bulle parallele
    :return: affiche le tableau trie
    """
    root = 0  # processus principale
    if RANK == root:
        numbers = np.genfromtxt("array.csv", delimiter=",")
        tabs = np.array_split(numbers, SIZE)
        for i in range(1, SIZE):
            data = {"tab": tabs[i]}
            send(data, dest=i)
        tab = tabs[root]
    else:
        data = recv(source=root)
        tab = data["tab"]
    tab.sort()
    for e in range(SIZE):
        if e % 2 == 0:
            if RANK % 2 == 0:
                tab = echange_min(RANK + 1, tab)
            else:
                tab = echange_max(RANK - 1, tab)
        else:
            if RANK % 2 == 0:
                tab = echange_max(RANK - 1, tab)
            else:
                tab = echange_min(RANK + 1, tab)
    if RANK == root:
        array = list(tab)
    for i in range(1, SIZE):
        if RANK == root:
            data = recv(source=i)
            array.extend(list(data["tab"]))
        else:
            data = {"tab": tab}
            send(data, dest=root)
    if RANK == root:
        print("tableau non trie ==> ", numbers)
        print("tableau apres tri ==> ", np.array(array))


if __name__ == '__main__':
    sort()
