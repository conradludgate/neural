#include "bind.h"
#include <iostream>

NN::NN() {
	nn.random();
}

void NN::Train(float lr, float input[784], float target[10])
{
	T_in i(input);
	T_out t(target);

	nn.train(lr, i, t);
}

void NN::Predict(float input[784], float output[10])
{
	Eigen::Map<T_out>(output, 10, 1) = nn.feedforward(T_in(input));
}