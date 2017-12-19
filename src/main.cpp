#include <iostream>

#include "int.hpp"

int main()
{
	AddTrainer add;
	add.train(0.5);

	std::cout << "5 + 4 = " << add.process({5.0, 4.0})[0] << std::endl;
}