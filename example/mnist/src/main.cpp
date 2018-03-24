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
	neural::Trainer<IMAGE_SIZE, 30, 10> mnist;
	mnist.Random();	

	load_data();

	float lr = 0.1;
	std::cout << "> Learning Rate: ";
	std::cin >> lr;
	mnist.set_lr(lr);

	float score = 0;
	int epochs = 0;
	while (score < 0.99)
	{
		prepare_training_data();
		train_epoch(mnist);

		prepare_testing_data();
		score = test(mnist);

		std::cout << "Trained for " << ++epochs << " epochs. " 
				<< "Score = " << 100*score << "%" << std::endl;
	}
}