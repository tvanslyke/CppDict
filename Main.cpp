#include "Dict.h"
#include <string>
#include <iostream>
#include <unordered_map>
using namespace std::literals;
int main()
{
	HashDict<std::string, int> hd;
	hd["howdy"] = 1;
	hd["there"] = 2;
	hd["pardner"] = 3;
	hd["there"] = 4;
	hd["pardner"] = 5;
	hd["test1"] = 5;
	hd["fest2"] = 5;
	hd["tsst3"] = 5;
	hd["tsfg"] = 5;
	hd["bxcvb"] = 5;
	hd["testst"] = 5;
	hd["ddddd"] = 5;
	hd["c"] = 5;
	for(auto [word, id]:hd)
		std::cout << std::string(word) << ", " << int(id) << std::endl;
	std::cout << "Done" << std::endl;
	return 0;
}

