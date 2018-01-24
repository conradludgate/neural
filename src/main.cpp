#include <iostream>
#include <string>

#include "mnist/mnist.hpp"

int main()
{
	MNISTTrainer mnist;
	mnist.Zero();

	float p = 5E-5;

	std::cout << "Learning Rate: " << p << std::endl;

	mnist.train(1e-6, p);

	int index = rand() % 60000;
	mnist.set_index(index);
	auto data = mnist.dataf();

	auto output = mnist.process(data.input);
	int big = 0;
	for (int i = 0; i < 10; i++)
	{
		if (output[i] > output[big])
			big = i;
	}

	std::cout << "Image #" << index << " is classified as a " << big << " with " << output[big] << "\% likelihood" << std::endl;

	for (int i = 0; i < 10; i++)
	{
		if (data.expected[i] > data.expected[big])
			big = i;
	}

	std::cout << "Expected " << big << std::endl;
}