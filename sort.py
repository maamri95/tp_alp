import numpy as np
from mpi4py import MPI

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()  # numbre de processus
RANK = COMM.Get_rank()  # le processus actif
send = COMM.send  # fonction d'envoi
recv = COMM.recv  # fonction de reception
scatter = COMM.scatter  # fonction permetant de diffuse le tableau
gather = COMM.gather  # fonction qui permet de rassemble les resultat


def echange_min(me, tab):
    """
        recupere le n min des valeurs entre le la partie de process RANK et le process me,
        n etant la taille de la partie de RANK
        :param me: process adjacent a RANK
        :param tab: partie de process RANK
        :return: la nouvelle partie de process RANK
    """
    if SIZE > me >= 0:
        data = recv(source=me)
        tab2 = data["tab2"]
        data = {"tab2": tab}
        send(data, dest=me)
        z = np.concatenate((tab, tab2))
        z.sort()
        n = tab.size
        return z[:n]
    return tab


def echange_max(me, tab):
    """
    recupere le n max des valeurs entre le la partie de process RANK et le process me,
    n etant la taille de la partie de RANK
    :param me: process adjacent a RANK
    :param tab: partie de process RANK
    :return: la nouvelle partie de process RANK
    """
    if SIZE > me >= 0:
        data = {"tab2": tab}
        send(data, dest=me)
        data = recv(source=me)
        tab2 = data["tab2"]
        z = np.concatenate((tab, tab2))
        z.sort()
        n = tab.size
        return z[z.size - n:]
    return tab


def sort():
    """
    tri a bulle parallele
    :return: affiche le tableau trie
    """
    if RANK == 0:
        numbers = np.genfromtxt("array.csv", delimiter=",")
        tabs = np.array_split(numbers, SIZE)
        print("tableau non trie ==> ", numbers)
    else:
        tabs = None
    tab = scatter(tabs, root=0)
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
    array = gather(tab, root=0)
    if RANK == 0:
        tableau = []
        for i in range(len(array)):
            tableau.extend(array[i].tolist())
        print("tableau apres tri ==> ", np.array(tableau))


if __name__ == '__main__':
    sort()
