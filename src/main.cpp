#include <iostream>

#include "int.hpp"

int main()
{
	// auto add = neural::Net({2, 1});
	//add.setSize({2, 1});

	auto add = AddTrainer();

	add.train(0.25, 0.000001);

	// Int stores a single integer value
	// But also provides an iterator to 'loop' over that one value
	// This is a pretty useless example but works as a proof of concept
	Int n(0);

	// Since we are using our own output type, we initialise beforehand
	// and parse a reference
	add.process<int, Int>({359, 851}, n);

	std::cout << n << std::endl;
}