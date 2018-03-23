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

// template<ui A, ui B>
// struct TestData
// {
// 	vec<A> input;
// 	vec<B> expected;
// };

template<int First, int Second, int... Further>
class Trainer: public Net<First, Second, Further...>
{

static const int last = get_last<Second, Further...>();

public:

	void train(vec<First> input, vec<get_last<Second, Further...>()> expected)
	{
		back_prop<0, First, Second, Further...>(input, expected);
	}

	void set_lr(float lr)
	{
		learning_rate = lr;
	}

// 	void train(float tol, float p)
// 	{
// 		assert(tol > 0);
// 		//assert(p > 0);

// 		float cost = 0;
// 		while (average >= tol)
// 		{
// 			// Get test data
// 			auto td = dataf();

// 			// Perform back propagation
// 			back_prop<0, First, Second, Further...>(td.input, td.expected, p, cost);
// 			average = (average + cost) / 2;
// 			DEBUG_LOG(cost);
// 		}
// 	}

// protected:
// 	virtual TestData<First, last> dataf() = 0;

private:
	template<int n, int A, int B, int... Is>
	vec<A> back_prop(vec<A> input, vec<get_last<B, Is...>()> expected)
	{
		//vec<B> output = relu<B>(std::get<n>(this->m_weights) * input + std::get<n>(this->m_biases));
		vec<B> output = std::get<n>(this->m_weights) * input + std::get<n>(this->m_biases);

		vec<B> scale;

		if constexpr(sizeof...(Is) != 0)
		{
			scale = learning_rate * 2 * (back_prop<n+1, B, Is...>(output, expected) - output);
		} else {
			scale = learning_rate * 2 * (expected - output);
		}
		
		auto d_input = std::get<n>(this->m_weights).transpose() * scale; // AxB * Bx1 = Ax1
		std::get<n>(this->m_weights) += scale * input.transpose(); // BxA = Bx1 * 1xA
		std::get<n>(this->m_biases) += scale;

		return input + d_input;
	}

	//float average = 1;
	float learning_rate = 0.1;
};

}