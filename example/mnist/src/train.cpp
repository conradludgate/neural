#include <fstream>
#include <cassert>
#include <iostream>
#include <cstdlib>

#include "neural/train.hpp"

const int IMAGE_SIZE = 28*28;

std::ifstream m_train_images;
std::ifstream m_train_labels;

Eigen::Matrix<float, IMAGE_SIZE, 1> images[60000];
int labels [60000];

void get_images()
{
	m_train_images.seekg(16, std::ios::beg);

	for (int image = 0; image < 60000; image++)
	{
		char * buffer = new char [IMAGE_SIZE];
		m_train_images.read(buffer, IMAGE_SIZE);

		for (int i = 0; i < IMAGE_SIZE; i++)
		{
			images[image][i] = (float)buffer[i] / 128.0;
		}
	}
}

void get_labels()
{
	m_train_labels.seekg(8, std::ios::beg);

	for (int label = 0; label < 60000; label++)
	{
		unsigned char l;
		m_train_labels >> l;

		labels[label] = l;
	}
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
		mnist.Zero();
	}

	if (argc >= 1)
	{
		float lr = strtof(argv[0], nullptr);
		mnist.set_lr(lr > 0 ? lr : 0.1);
	}

	int epochs
	if (argc >= 2)
	{
		epochs = atoi(argv[1]);
	}

	std::cout << "Loaded Network" << std::endl;
	std::cout << "Learning rate: " << mnist.get_lr() << std::endl;

	m_train_images.open("data/trainingdata", std::ios::binary | std::ios::in);
	m_train_labels.open("data/traininglabels", std::ios::binary | std::ios::in);

	if (m_train_images.fail() || m_train_labels.fail())
		exit(1);	

	get_images();
	get_labels();

	std::cout << "Loaded Data" << std::endl;

	for (int e = 0; e < epochs; e++)
	{
		std::cout << "Epoch " << e << std::endl;
		for  (int index = 0; index < 60000; index++)
		{
			Eigen::Matrix<float, 10, 1> expected = Eigen::Matrix<float, 10, 1>::Zero();
			expected[labels[index]%10] = 1;

			mnist.train(images[index], expected);
		}
	}

	std::cout << "Training Complete" << std::endl;

	// Save network to a file
	mnist.Save("networks/mnist");
}