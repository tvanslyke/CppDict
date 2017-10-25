
dict: Main.o
	g++ -std=c++17 -g -O0 -fno-omit-frame-pointer Main.o -o dict

Main.o: Dict.h HashUtils.h
	g++ -std=c++17 -g -O0 -fno-omit-frame-pointer -c Main.cpp

clean:
	rm Main.o dict
