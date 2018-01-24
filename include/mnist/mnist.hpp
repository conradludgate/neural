#include "neural/train.hpp"
#include <iostream>
#include <fstream>

const int IMAGE_SIZE = 28*28;

class MNISTTrainer: public neural::Trainer<IMAGE_SIZE, 16, 16, 10>
{

public:
	MNISTTrainer()
	{
		m_train_images.open("include/mnist/imagedata", std::ios::binary | std::ios::in);
		m_train_labels.open("include/mnist/imagelabels", std::ios::binary | std::ios::in);

		if (m_train_images.fail() || m_train_labels.fail())
			exit(1);
	}

	neural::TestData<IMAGE_SIZE, 10> dataf()
	{
		if (m_train_images.fail() || m_train_labels.fail())
			exit(1);

		neural::TestData<IMAGE_SIZE, 10> td;

		m_train_images.seekg(16 + IMAGE_SIZE * m_index, std::ios::beg);
		char * buffer = new char [IMAGE_SIZE];

		m_train_images.read(buffer, IMAGE_SIZE);

		for (int i = 0; i < 28 * 28; i++)
		{
			td.input[i] = (uint8_t)buffer[i] / 256.0;
		}

		char i = 0;
		m_train_labels.seekg(8 + m_index, std::ios::beg);
		m_train_labels >> i;

		td.expected.setZero();
		td.expected[i % 10] = 1;

		m_index = (m_index + 1) % 60000;

		return td;
	}

	void set_index(int index)
	{
		m_index = index % 60000;
	}

private:
	std::ifstream m_train_images;
	std::ifstream m_train_labels;

	int m_index = 0;
};