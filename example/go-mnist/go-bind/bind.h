#ifndef MNIST_H
#define MNIST_H

#include "neural/net.hpp"
#include "neural/layer.hpp"

typedef neural::Net<
			neural::Layer<784, 30>,
			neural::Layer<30, 10>> net;

typedef vec<784> T_in;
typedef vec<10> T_out;

class NN
{
public:
	NN();
	void Train(float lr, float input[784], float target[10]);
	void Predict(float input[784], float output[10]);

private:
	net nn;
};

#endif