#ifndef IMAGE_SIZE
#define IMAGE_SIZE 784
#endif

#include <neural/train.hpp>

vec<IMAGE_SIZE> get_image(int index);
int get_label(int index);

void train_epoch(neural::Trainer<IMAGE_SIZE, 30, 10>& nn)
{
	for  (int index = 0; index < 60000; index++)
	{
		Eigen::Matrix<float, 10, 1> expected = Eigen::Matrix<float, 10, 1>::Zero();
		expected[get_label(index)%10] = 1;

		nn.train(get_image(index), expected);
	}
}