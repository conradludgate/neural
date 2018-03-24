#ifndef IMAGE_SIZE
#define IMAGE_SIZE 784
#endif

#include <neural/train.hpp>

vec<IMAGE_SIZE> get_image(int index);
int get_label(int index);

// Train for one epoch (Over all the training data available)
void train_epoch(neural::Trainer<IMAGE_SIZE, 30, 10>& nn)
{
	for  (int index = 0; index < 60000; index++)
	{
		// Process the expected value into a vector
		Eigen::Matrix<float, 10, 1> expected = Eigen::Matrix<float, 10, 1>::Zero();
		expected[get_label(index)%10] = 1;

		// Train
		nn.train(get_image(index), expected);
	}
}