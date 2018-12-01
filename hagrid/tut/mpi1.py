
# from https://mpi4py.scipy.org/docs/usrman/tutorial.html
import time

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def test1(): 
    if rank == 0:
        data = {'a': 7, 'b': 3.14}
        comm.send(data, dest=1, tag=11)
    elif rank == 1:
        data = comm.recv(source=0, tag=11)
        print(data)

def test2():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        data = {'a': 7, 'b': 3.14}
        req = comm.isend(data, dest=1, tag=11)
        req.wait()
    elif rank == 1:
        req = comm.irecv(source=0, tag=11)
        data = req.wait()
        print(data)
    else:
        print(rank)

def test3():
    oComm = MPI.COMM_WORLD
    nRank = comm.Get_rank()
    nSize = comm.Get_size()

    if nRank == 0:
        data = [(i+1)**2 for i in range(nSize)]
    else:
        data = None

    print ("before", nRank, type(data), data)
    data = comm.scatter(data, root=0)

    assert data == (rank+1)**2
    print("after", nRank, type(data), data)
    #print("sleeping 60 to keep process running")
    #time.sleep(60)

def test4(): 
    oComm = MPI.COMM_WORLD
    nRank = comm.Get_rank()
    nSize = comm.Get_size()

    if nRank == 0:
        data = [(i+1)**2 for i in range(nSize)]
    else:
        data = None

    print ("before", nRank, type(data), data)
    data = comm.scatter(data, root=0)

    assert data == (rank+1)**2
    print("after", nRank, type(data), data)
    #print("sleeping 60 to keep process running")
    #time.sleep(60)


if __name__=="__main__":
    test3()