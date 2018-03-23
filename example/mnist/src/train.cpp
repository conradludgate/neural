#include <fstream>
#include <cassert>
#include <iostream>
#include <cstdlib>

#include "neural/train.hpp"

const int IMAGE_SIZE = 28*28;

std::ifstream m_train_images;
std::ifstream m_train_labels;

Eigen::Matrix<float, IMAGE_SIZE, 1> get_input()
{
	Eigen::Matrix<float, IMAGE_SIZE, 1> input;

	char * buffer = new char [IMAGE_SIZE];
	m_train_images.read(buffer, IMAGE_SIZE);

	for (int i = 0; i < IMAGE_SIZE; i++)
	{
		input[i] = (float)buffer[0] / 255.0;
	}

	return input;
}

int get_output(Eigen::Matrix<float, 10, 1> output)
{
	int big = 0;
	for (int i = 0; i < 10; i++)
	{
		if (output[i] > output[big])
			big = i;
	}

	return big;
}

int get_label()
{
	char label;
	m_train_labels >> label;

	return (int)(10 + label) % 10;
}

int main(int argc, char *argv[])
{
	neural::Trainer<IMAGE_SIZE, 40, 40, 10> mnist;

	try
	{
		mnist.Load("networks/mnist");
	}
	catch (const boost::archive::archive_exception e)
	{
		mnist.Random();
	}

	if (argc == 1)
	{
		float lr = strtof(argv[0], nullptr);
		mnist.set_lr(lr > 0 ? lr : 0.1);
	}

	std::cout << "Loaded Network" << std::endl;

	m_train_images.open("data/trainingdata", std::ios::binary | std::ios::in);
	m_train_labels.open("data/traininglabels", std::ios::binary | std::ios::in);

	if (m_train_images.fail() || m_train_labels.fail())
		exit(1);

	m_train_images.seekg(16, std::ios::beg);
	m_train_labels.seekg(8, std::ios::beg);

	std::cout << "Loaded Data" << std::endl;

	for (int x = 0; x < 600; x++)
	{
		switch (x % 4)
		{
			case 0:
				printf("\\\r");
				break;
			case 1:
				printf("|\r");
				break;
			case 2:
				printf("/\r");
				break;
			case 3:
				printf("|\r");
				break;
		}

		for (int index = 0; index < 100; index++)
		{
			Eigen::Matrix<float, 10, 1> expected;
			expected.setZero();

			int label = get_label();
			assert(0 <= label < 10);

			expected[get_label()] = 1;

			mnist.train(get_input(), expected);
		}
	}

	std::cout << "Training Complete" << std::endl;

	// Save network to a file
	mnist.Save("networks/mnist");
}