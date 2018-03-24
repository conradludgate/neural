#ifdef DEBUG
#define DEBUG_LOG(...) printf (__VA_ARGS__)
#else
#define DEBUG_LOG(...)
#endif

#pragma once

#include "neural/net.hpp"
#include <vector>
#include <cstdio>

namespace neural
{

template<int First, int Second, int... Further>
class Trainer: public Net<First, Second, Further...>
{

static const int last = get_last<Second, Further...>();

public:

	float train(vec<First> input, vec<last> expected)
	{
		auto delta = back_prop<0, First, Second, Further...>(input, expected);
		return delta.dot(delta);
	}

	void set_lr(float lr)
	{
		learning_rate = lr;
	}

	float get_lr()
	{
		return learning_rate;
	}

private:
	template<int n, int A, int B, int... Is>
	vec<A> back_prop(vec<A> input, vec<last> expected)
	{
		vec<B> output = g<B>(std::get<n>(this->m_weights) * input + std::get<n>(this->m_biases));
		vec<B> scale;

		if constexpr(sizeof...(Is) != 0)
		{
			scale = learning_rate * back_prop<n+1, B, Is...>(output, expected);
		} else {
			scale = learning_rate * (output - expected);
		}

		vec<B>& bias = std::get<n>(this->m_biases);
		mat<B, A>& weight = std::get<n>(this->m_weights);

		gprime(scale, output);
		vec<A> dinput = weight.transpose() * scale;

		for (int i = 0; i < B; i++)
		{
			bias[i] -= scale[i];
			weight.row(i) -= scale[i] * input;
		}

		return dinput;
	}

	float learning_rate = 0.1;
};

}

