#ifndef IMAGE_SIZE
#define IMAGE_SIZE 784
#endif

#include <iostream>

#include "neural/net.hpp"

typedef neural::Net<IMAGE_SIZE, 10,
			neural::NeuronLayer<IMAGE_SIZE, 30>,
			neural::NeuronLayer<30, 10>> NN;

void load_data();
void prepare_training_data();
void prepare_testing_data();

float train_epoch(NN& nn, float lr);
float test(NN& nn);

int main(int argc, char *argv[])
{
	// Create and initialise the network
	NN mnist;
	mnist.Random();	

	// Load the training and testing data
	load_data();

	// Set the learning rate for the network (default 0.1)
	float lr = 0.1;
	std::cout << "> Learning Rate: ";
	std::cin >> lr;

	// Train until satisfied.
	float score = 0;
	int epochs = 0;	
	while (score < 0.99)
	{
		// Train
		prepare_training_data();
		train_epoch(mnist, lr);

		// Test
		prepare_testing_data();
		score = test(mnist);

		// Output each step of our training, so we know it's still running
		// And to gauge how good our learning rate is
		std::cout << "Trained for " << ++epochs << " epochs. " 
				<< "Score = " << 100*score << "%" << std::endl;
	}

	// TODO: Serialization of the network. Store into a file for use elsewhere.
	// Tried boost::serialization, had some bugs
}