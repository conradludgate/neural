#pragma once

#include "neural/net.hpp"
#include <array>
#include <vector>
#include <iostream>

#include <chrono>
#include <thread>

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
	void train(float tol)
	{
		float cost = tol;

		while (cost >= tol)
		{
			auto td = dataf();
			back_prop<0, First, Second, Further...>(td.input, td.expected, cost);
			
			//std::cout << cost << std::endl;
			//std::cout << std::endl;

			//std::this_thread::sleep_for(std::chrono::milliseconds(250));
		}
	}

protected:
	virtual TestData<First, last> dataf() = 0;

private:
	template<ui n, ui A, ui B, ui... Is>
	vec<A> back_prop(vec<A> input, vec<get_last<B, Is...>()> expected, float& cost)
	{
		vec<B> output = std::get<n>(this->m_weights) * input + std::get<n>(this->m_biases);

		if constexpr(sizeof...(Is) != 0)
		{
			return back_prop<n, A, B>(input, back_prop<n+1, B, Is...>(output, expected));
		}

		if constexpr(last == B) {
			auto diff = expected - output;
			cost = diff.dot(diff);
		}

		float p = 0.000001;

		//auto scale = relu(2 * (expected - output));
		auto scale = 2 * (expected - output);
		auto d_input = p * std::get<n>(this->m_weights).transpose() * scale;
		std::get<n>(this->m_weights) += p * scale * input.transpose();
		std::get<n>(this->m_biases) += p * scale;

		return input + d_input;
	}
};

}