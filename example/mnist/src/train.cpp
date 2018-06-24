#ifndef IMAGE_SIZE
#define IMAGE_SIZE 784
#endif

#include <utility>

#include <neural/net.hpp>
#include <neural/layer.hpp>

typedef neural::Net<
			neural::Layer<IMAGE_SIZE, 30>,
			neural::Layer<30, 10>> NN;

int get_output(vec<10> output);
vec<IMAGE_SIZE> get_image(int index);
int get_label(int index);

// Train for one epoch (Over all the training data available)
void train_epoch(NN& nn, float lr)
{
	for  (int index = 0; index < 60000; index++)
	{
		// Process the expected value into a vector
		vec<10> expected = vec<10>::Zero();
		expected[get_label(index)%10] = 1;

		// Train
		nn.train(lr, get_image(index), expected);
	}
}

// Test all of our test data, generate a score
float test(NN& nn)
{
	int score = 0;

	for (int index = 0; index < 10000; index++)
	{
		// Generate an output and make it more accesible
		int output = get_output(nn.feedforward(
			std::forward<vec<IMAGE_SIZE>>(get_image(index))));

		// If the output is what we expected, increment the score
		if (output == get_label(index))
		{
			score++;
		}
	}

	return (float)score / 10000.0;
}