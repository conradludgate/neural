#pragma once

#include "neural/net.hpp"
#include <array>
#include <vector>
#include <iostream>

namespace neural
{

template<ui A, ui B>
struct TestData
{
	vec<A> input;
	vec<B> expected;
};

template<ui A, ui B, ui... Is>
struct Neurons
{
	decltype(make_weights<A, B, Is...>()) weights;
	decltype(make_biases<B, Is...>()) biases;
};

template<ui First, ui Second, ui... Further>
class Trainer: public Net<First, Second, Further...>
{

static const ui last = get_last<Second, Further...>();

public:
	void train(float tol)
	{
		Neurons<First, Second, Further...> deltas;

		float cost = tol;

		while (cost >= tol)
		{
			zero_neurons<0>(deltas);
			cost = 0;

			for (int i = 0; i < 10000; i++)
			{
				auto td = dataf();
				auto d_input = back_prop<0, First, Second, Further...>(td.input, td.expected, deltas);
				cost += d_input.dot(d_input);	
			}

			div_neurons<0>(deltas, 10000);
			cost /= 10000;

			std::cout << cost << std::endl;
		}
	}

protected:
	virtual TestData<First, last> dataf() = 0;

private:
	template<ui Nodes>
	float costf(mat<Nodes, 1> o, mat<Nodes, 1> e)
	{
		auto diff = e - o;
		return diff.dot(diff);
	}

	float costf(mat<last, 1> output, mat<last, 1> expected)
	{
		costf<last>(output, expected);
	}

	// Refer to 3b1b
	template<ui n, ui A, ui B, ui... Is>
	vec<A> back_prop(vec<A> input, vec<get_last<B, Is...>()> expected, Neurons<First, Second, Further...>& deltas)
	{
		vec<B> output = std::get<n>(this->m_weights) * input + std::get<n>(this->m_biases);

		//for (int i = 0; i < B; ++i)
		//	if (output[i] < 0)
		//		output[i] = 0;

		if constexpr(sizeof...(Is) != 0)
		{
			return back_prop<n, A, B>(input, back_prop<n+1, B, Is...>(output, expected));
		}

		// relu(x) = x if x > 0 else 0
		// relu'(x) = 1 if x > 0 else 0

		// cost = (output - expected) ^ 2
		// output = relu(z)
		// z = weight * input + bias
 
		// d cost / d weight = (d cost / d output) * (d output / d z) * (d z / d weight)
		// = 2(output - expected) * relu'(z) * input
		// d cost / d bias = 2(output - expected) * relu'(z)
		// d cost / d input = 2(output - expected) * relu'(z) * weight

		// New expected = input - (d cost/ d input)

		vec<A> d_input;
		// vec<B> d_bias;
		// mat<B, A> d_weight;
		d_input.setZero();
		// d_bias.setZero();
		// d_weight.setZero();

		//std::get<n>(deltas.weights).setZero();
		//std::get<n>(deltas.biases).setZero();

		for (int i = 0; i < A; ++i)
		{
			for (int j = 0; j < B; ++j)
			{
				if (output[j] > 0)
				{
					// twooe = -2(output - expected). => Add the deltas instead of subtract.
					float twooe = 2 * (expected[j] - output[j]);
					d_input[i] += twooe * std::get<n>(this->m_weights)(j, i);
					std::get<n>(deltas.weights)(j, i) += twooe * input[i];
					std::get<n>(deltas.biases)(j) += twooe;
				}
			}
		}

		// std::cout << "m_weight: " << std::get<n>(this->m_weights) << std::endl;
		// std::cout << "m_bias: " << std::get<n>(this->m_biases) << std::endl;

		// std::cout << "input: " << input << std::endl;
		// std::cout << "output: " << output << std::endl;
		// std::cout << "expected: " << expected << std::endl;

		// // BUG: Stop the weights/biases exploding to inf
		// std::cout << "d_weight: " << d_weight << std::endl;
		// std::cout << "d_bias: " << d_bias / A << std::endl;

		// std::get<n>(this->m_weights) += d_weight;
		// std::get<n>(this->m_biases) += d_bias / A;

		return input + (d_input / B);
	}

	template<ui n>
	void zero_neurons(Neurons<First, Second, Further...>& neurons)
	{
		//std::get<n>(m_weights) = Eigen::Matrix<float, B, A>::Zero();
		std::get<n>(neurons.weights).setZero();
		std::get<n>(neurons.biases).setZero();

		if constexpr(sizeof...(Further) != n)
		{
			zero_neurons<n+1>(neurons);
		}
	}

	template<ui n>
	void div_neurons(Neurons<First, Second, Further...>& neurons, int count)
	{
		//std::get<n>(m_weights) = Eigen::Matrix<float, B, A>::Zero();
		std::get<n>(neurons.weights) /= float(count);
		std::get<n>(neurons.biases) /= float(count);

		if constexpr(sizeof...(Further) != n)
		{
			div_neurons<n+1>(neurons);
		}
	}
};

}