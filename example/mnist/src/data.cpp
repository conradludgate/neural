/// data.cpp
/// loads the mnist databases
/// also processes the data to be used by the nn

#ifndef IMAGE_SIZE
#define IMAGE_SIZE 784
#endif

#include <fstream>
#include <Eigen/Dense>
#include <iostream>

template<int A>
using vec = Eigen::Matrix<float, A, 1>;

std::ifstream train_images;
std::ifstream train_labels;
std::ifstream test_images;
std::ifstream test_labels;

std::ifstream* images = nullptr;
std::ifstream* labels = nullptr;

void load_data()
{
	train_images.open("data/trainingdata", std::ios::binary | std::ios::in);
	train_labels.open("data/traininglabels", std::ios::binary | std::ios::in);
	test_images.open("data/testingdata", std::ios::binary | std::ios::in);
	test_labels.open("data/testinglabels", std::ios::binary | std::ios::in);

	if (train_images.fail() || train_labels.fail() 
		|| test_images.fail() || test_labels.fail())
		exit(1);
}

void prepare_training_data()
{
	images = &train_images;
	labels = &train_labels;
}

void prepare_testing_data()
{
	images = &test_images;
	labels = &test_labels;
}

// Turns the nn output into a single value
int get_output(vec<10> output)
{
	int big = 0;
	for (int i = 0; i < 10; i++)
	{
		if (output[i] > output[big])
			big = i;
	}

	return big;
}

// Create a vector type to enter into the network
vec<IMAGE_SIZE> get_image(int index)
{
	images->seekg(16 + IMAGE_SIZE*index, std::ios::beg);

	vec<IMAGE_SIZE> image;
	uint8_t buffer [IMAGE_SIZE];
	images->read((char*) buffer, IMAGE_SIZE);

	for (int i = 0; i < IMAGE_SIZE; i++)
	{
		image[i] = (float)buffer[i] / 255.0;
	}

	return image;
}

// What value is expected
int get_label(int index)
{
	labels->seekg(8 + index, std::ios::beg);

	char x;
	labels->read(&x, 1);
	return x;
}