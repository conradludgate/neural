#ifndef IMAGE_SIZE
#define IMAGE_SIZE 784
#endif

#include <iostream>

#include "neural/train.hpp"

void load_data();
void prepare_training_data();
void prepare_testing_data();

float test(neural::Net<IMAGE_SIZE, 30, 10>& nn);
void train_epoch(neural::Trainer<IMAGE_SIZE, 30, 10>& nn);

int main(int argc, char *argv[])
{
	// Create and initialise a trainer network
	neural::Trainer<IMAGE_SIZE, 30, 10> mnist;
	mnist.Random();	

	// Load the training and testing data
	load_data();

	// Set the learning rate for the network (default 0.1)
	float lr = 0.1;
	std::cout << "> Learning Rate: ";
	std::cin >> lr;
	mnist.set_lr(lr);

	// Train until satisfied.
	float score = 0;
	int epochs = 0;	
	while (score < 0.99)
	{
		// Train
		prepare_training_data();
		train_epoch(mnist);

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