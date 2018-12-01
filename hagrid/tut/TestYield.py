

def testyield(nLen):
    for i in range(nLen):
        yield i**2

def main():

    iTest = testyield(100)
    while True:
        nVal = iTest.__next__() 
        print(nVal)
        if nVal > 100:
            break
        

if __name__=="__main__":
    main()