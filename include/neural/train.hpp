#pragma once

#include "neural/net.hpp"
#include <array>
#include <vector>
#include <iostream>
#include <cstdio>

namespace neural
{
	
template<ui A, ui B>
struct TestData
{
	vec<A> input;
	vec<B> expected;
};

template<ui First, ui Second, ui... Further>
class Trainer: public Net<First, Second, Further...>
{

static const ui last = get_last<Second, Further...>();

public:
	void train(float tol, float p)
	{
		float cost = tol;

		while (cost >= tol)
		{
			// Get test data
			auto td = dataf();

			// Perform back propagation
			back_prop<0, First, Second, Further...>(td.input, td.expected, p, cost);
		}
	}

protected:
	virtual TestData<First, last> dataf() = 0;

private:
	template<ui n, ui A, ui B, ui... Is>
	vec<A> back_prop(vec<A> input, vec<get_last<B, Is...>()> expected, float p, float& cost)
	{
		vec<B> output = relu<B>(std::get<n>(this->m_weights) * input + std::get<n>(this->m_biases));

		vec<B> e;
		if constexpr(sizeof...(Is) != 0)
		{
			e = back_prop<n+1, B, Is...>(output, expected, p, cost);
		} else {
			e = expected;

			auto diff = expected - output;
			cost = diff.dot(diff);
		}
		
		auto scale = relu<B>(2 * (e - output));
		auto d_input = p * std::get<n>(this->m_weights).transpose() * scale; // AxB * Bx1 = Ax1
		std::get<n>(this->m_weights) += p * scale * input.transpose(); // BxA = Bx1 * 1xA
		std::get<n>(this->m_biases) += p * scale; // 

		return input + d_input;
	}
};

}