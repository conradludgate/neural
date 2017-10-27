#include "neural/net.hpp"
#include <vector>
#include <iostream>
#include <array>

#include "int.hpp"

int main()
{
	auto add = neural::Net({2, 1});

	// Int stores a single integer value
	// But also provides an iterator to 'loop' over that one value
	// This is a pretty useless example but works as a proof of concept
	Int n(0);

	// Since we are using our own output type, we initialise beforehand
	// and parse a reference
	add.process<int, Int>({5, 10}, n);

	std::cout << n << std::endl;
}