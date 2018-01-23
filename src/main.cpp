#include <iostream>

#include "int.hpp"

int main()
{
	AddTrainer add;
	add.train(1e-12);

	auto data = add.dataf();
	int output = add.process(data.input)[0];

	std::cout << data.input[0] << " + " << data.input[1] << " = " << output << std::endl;
}