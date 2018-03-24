#ifndef IMAGE_SIZE
#define IMAGE_SIZE 784
#endif

#include <neural/net.hpp>

int get_output(vec<10> output);
vec<IMAGE_SIZE> get_image(int index);
int get_label(int index);

// Test all of our test data, generate a score
float test(neural::Net<IMAGE_SIZE, 30, 10>& nn)
{
	int score = 0;

	for (int index = 0; index < 10000; index++)
	{
		// Generate an output and make it more accesible
		int output = get_output(nn.predict(get_image(index)));

		// If the output is what we expected, increment the score
		if (output == get_label(index))
		{
			score++;
		}
	}

	return (float)score / 10000.0;
}