#include <neural/net.hpp>

const int IMAGE_SIZE = 28*28;

int get_output(vec<10> output);
vec<IMAGE_SIZE> get_image(int index);
int get_label(int index);

float test(neural::Net<IMAGE_SIZE, 30, 10>& nn)
{
	int score = 0;

	for (int index = 0; index < 10000; index++)
	{
		int output = get_output(nn.process(get_image(index)));

		if (output == get_label(index))
		{
			score++;
		}
	}

	return (float)score / 10000.0;
}