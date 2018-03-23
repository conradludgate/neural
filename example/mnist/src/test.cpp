#include <iostream>
#include <fstream>

#include "neural/net.hpp"

const int IMAGE_SIZE = 28*28;

std::ifstream m_test_images;
std::ifstream m_test_labels;

Eigen::Matrix<float, IMAGE_SIZE, 1> get_input()
{
	Eigen::Matrix<float, IMAGE_SIZE, 1> input;

	char * buffer = new char [IMAGE_SIZE];
	m_test_images.read(buffer, IMAGE_SIZE);

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
	m_test_labels >> label;

	return (int)(10 + label) % 10;
}

int main()
{
	neural::Net<IMAGE_SIZE, 40, 40, 10> mnist;

	try
	{
		mnist.Load("networks/mnist");
	}
	catch (const boost::archive::archive_exception e)
	{
		mnist.Zero();
	}

	m_test_images.open("data/testingdata", std::ios::binary | std::ios::in);
	m_test_labels.open("data/testinglabels", std::ios::binary | std::ios::in);

	if (m_test_images.fail() || m_test_labels.fail())
		exit(1);

	m_test_images.seekg(16, std::ios::beg);
	m_test_labels.seekg(8, std::ios::beg);

	int score = 0;

	for (int index = 0; index < 10000; index++)
	{
		int output = get_output(mnist.process(get_input()));
		int label = get_label();

		if (output == label)
		{
			score++;
		}
	}

	// Display results
	std::cout << "Score: " << score << " / 10000" << std::endl;
}