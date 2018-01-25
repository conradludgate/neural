#include "neural/train.hpp"
#include <iostream>
#include <fstream>

const int IMAGE_SIZE = 28*28;

class MNISTTrainer: public neural::Trainer<IMAGE_SIZE, 40, 20, 10>
{

public:
	MNISTTrainer()
	{
		this->Zero();

		m_train_images.open("data/imagedata", std::ios::binary | std::ios::in);
		m_train_labels.open("data/imagelabels", std::ios::binary | std::ios::in);

		if (m_train_images.fail() || m_train_labels.fail())
			exit(1);

		set_index(0);
	}

	neural::TestData<IMAGE_SIZE, 10> dataf()
	{
		if (m_train_images.fail() || m_train_labels.fail())
			exit(1);

		neural::TestData<IMAGE_SIZE, 10> td;

		char * buffer = new char [IMAGE_SIZE];
		m_train_images.read(buffer, IMAGE_SIZE);

		for (int i = 0; i < 28 * 28; i++)
		{
			td.input[i] = (uint8_t)buffer[i] / 256.0;
		}

		char i = 0;
		m_train_labels >> i;

		td.expected.setZero();
		td.expected[i % 10] = 1;

		return td;
	}

	void set_index(int index)
	{
		index %= 60000;
		m_train_images.seekg(16 + IMAGE_SIZE * index, std::ios::beg);
		m_train_labels.seekg(8 + index, std::ios::beg);
	}

private:
	std::ifstream m_train_images;
	std::ifstream m_train_labels;
};