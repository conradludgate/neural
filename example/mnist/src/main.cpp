#include <iostream>
#include <string>
#include <fstream>

#include "mnist.hpp"

int main()
{
	// Init and Zero trainer
	MNISTTrainer mnist;

	// Begin Training
	mnist.train(0.01, 1E-6);

	// Get random image
	int index = rand() % 60000;
	mnist.set_index(index);
	auto data = mnist.dataf();

	// Evalute network's guess
	auto output = mnist.process(data.input);
	int big = 0;
	for (int i = 0; i < 10; i++)
	{
		if (output[i] > output[big])
			big = i;
	}

	// Display results
	std::cout << "Image #" << index << " is classified as a " << big << " with " << output[big] << "\% likelihood" << std::endl;

	// What was expected
	for (int i = 0; i < 10; i++)
	{
		if (data.expected[i] > data.expected[big])
			big = i;
	}

	std::cout << "Expected " << big << std::endl;

	// Save network to a file
	mnist.Save("mnist-network");
}