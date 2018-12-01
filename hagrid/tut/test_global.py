

#abc=123

def fn1():
	global abc
	
	print ("abc exists: ", "abc" in globals())

	if not "abc" in globals():
		print("Creating abc in globals()")
		abc=456
	
	print ("abc exists: ", "abc" in globals())

	print(abc)
	print(globals())

def fn2():
	global abc
	print ("fn2 abc:", abc)

def main():
	fn1()
	fn2()

if __name__ == "__main__":
	main()

