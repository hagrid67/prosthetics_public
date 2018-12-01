import traceback


def f2():
    print("f2")
    raise ValueError("testing 123")
    


def f1():
    print("f1")
    f2()


def main():
    print("main")
    try:
        f1()
    except ValueError as oVE:
        print ("ValueError:", oVE)
        print (oVE.__traceback__)
        traceback.print_tb(oVE.__traceback__)

if __name__=="__main__":
    main()