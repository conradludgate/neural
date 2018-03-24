#include <fstream>
#include <cassert>
#include <iostream>
#include <cstdlib>
#include <stdio.h>

#include "neural/train.hpp"

#include <png++/png.hpp>

const int IMAGE_SIZE = 28*28;

neural::Trainer<IMAGE_SIZE, 30, 10> mnist;

std::ifstream* images;
std::ifstream* labels;

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

int get_label(int index)
{
	labels->seekg(8 + index, std::ios::beg);

	char x;
	labels->read(&x, 1);
	return x;
}

// void save_png(int index)
// {
// 	int label = get_label(index);
// 	vec<28*28> image_data = get_image(index);
// 	png::image< png::ga_pixel > image(28, 28);
// 	for (png::uint_32 y = 0; y < image.get_height(); ++y)
// 	{
// 	    for (png::uint_32 x = 0; x < image.get_width(); ++x)
// 	    {
// 	    	float colour = image_data[x + y*28] * 255.0;
// 	        image[y][x] = png::ga_pixel(255 - colour, 255);
// 	    }
// 	}

// 	char buffer [8 + 1 + 5 + 5 + 1 + 1 + 4];
//   	sprintf (buffer, "pictures/image%05d-%d.png", index, label);
// 	image.write(buffer);
// }

void train_epoch()
{
	for  (int index = 0; index < 60000; index++)
	{
		Eigen::Matrix<float, 10, 1> expected = Eigen::Matrix<float, 10, 1>::Zero();
		expected[get_label(index)%10] = 1;

		mnist.train(get_image(index), expected);
	}
}

float test()
{
	int score = 0;

	for (int index = 0; index < 10000; index++)
	{
		int output = get_output(mnist.process(get_image(index)));

		if (output == get_label(index))
		{
			score++;
		}
	}

	return (float)score / 10000.0;
}

int main(int argc, char *argv[])
{
	mnist.Random();

	std::ifstream train_images;
	std::ifstream train_labels;
	std::ifstream test_images;
	std::ifstream test_labels;

	train_images.open("data/trainingdata", std::ios::binary | std::ios::in);
	train_labels.open("data/traininglabels", std::ios::binary | std::ios::in);
	test_images.open("data/testingdata", std::ios::binary | std::ios::in);
	test_labels.open("data/testinglabels", std::ios::binary | std::ios::in);

	if (train_images.fail() || train_labels.fail() || test_images.fail() || test_labels.fail())
		exit(1);	

	std::cout << "Loaded Data" << std::endl;

	float lr = 0.1;
	std::cout << "> Learning Rate: ";
	std::cin >> lr;
	mnist.set_lr(lr);

	float score = 0;
	int epochs = 0;
	while (score < 0.99)
	{
		images = &train_images;
		labels = &train_labels;
		train_epoch();	

		images = &test_images;
		labels = &test_labels;
		score = test();

		std::cout << "Trained for " << ++epochs << " epochs. Score = " << 100*score << "%" << std::endl;
	}
}