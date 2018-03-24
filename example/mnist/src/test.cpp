#include <iostream>
#include <fstream>

#include "neural/net.hpp"

const int IMAGE_SIZE = 28*28;

std::ifstream m_test_images;
std::ifstream m_test_labels;

Eigen::Matrix<float, IMAGE_SIZE, 1> images[10000];
int labels [10000];

void get_images()
{
	m_test_images.seekg(16, std::ios::beg);

	for (int image = 0; image < 10000; image++)
	{
		char * buffer = new char [IMAGE_SIZE];
		m_test_images.read(buffer, IMAGE_SIZE);

		for (int i = 0; i < IMAGE_SIZE; i++)
		{
			images[image][i] = (float)buffer[i] / 128.0;
		}
	}
}

void get_labels()
{
	m_test_labels.seekg(8, std::ios::beg);

	for (int label = 0; label < 10000; label++)
	{
		unsigned char l;
		m_test_labels >> l;

		labels[label] = l;
	}
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

int main()
{
	neural::Net<IMAGE_SIZE, 40, 40, 10> mnist;

	try
	{
		mnist.Load("networks/mnist");
	}
	catch (const boost::archive::archive_exception e)
	{
		mnist.Random();
	}

	std::cout << "Loaded Network" << std::endl;

	m_test_images.open("data/testingdata", std::ios::binary | std::ios::in);
	m_test_labels.open("data/testinglabels", std::ios::binary | std::ios::in);

	if (m_test_images.fail() || m_test_labels.fail())
		exit(1);

	get_images();
	get_labels();

	std::cout << "Loaded Data" << std::endl;

	int score = 0;

	for (int index = 0; index < 10000; index++)
	{
		int output = get_output(mnist.process(images[index]));

		if (output == labels[index])
		{
			score++;
		}
	}

	// Display results
	std::cout << "Score: " << score << " / 10000" << std::endl;
}