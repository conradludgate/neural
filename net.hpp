#pragma once

#include <iostream>
#include <type_traits>

#include "neural/util.hpp"
#include "neural/cost_layer.hpp"
#include "neural/template_util.hpp"

namespace neural
{

template <
	typename S,
	typename C,
	typename InputLayer, typename... HiddenLayers>
class Net
{
private:
	std::tuple<InputLayer, HiddenLayers...> layers;

public:
	Net() {}
	Net(std::tuple<InputLayer, HiddenLayers...> layers) : layers(layers) {}

	using Scalar = S;
	using Cost = C;

	static const int Inputs = InputLayer::Inputs;
	static const int Outputs = std::tuple_element<
		sizeof...(HiddenLayers), decltype(layers)>::type::Outputs;
	static const int Size = utils::__get_size<InputLayer, HiddenLayers...>();
	static const int NumLayers = sizeof...(HiddenLayers) + 1;

	static const decltype(utils::__get_info<InputLayer, HiddenLayers...>()) info;

	void random() { random<0>(); }

	friend std::ostream &operator<<(std::ostream &os, const Net &net)
	{
		return net.save<0>(os);
	}
	friend std::istream &operator>>(std::istream &is, Net &net)
	{
		return net.load<0>(is);
	}

	// Train the network using a batch of inputs
	// and their corresponding expected values
	template <int Batch>
	void train(
		Scalar learning_rate,
		mat<Scalar, Inputs, Batch> input,
		mat<Scalar, Outputs, Batch> expected)
	{
		feedforward_backward(learning_rate, input, expected);
	}

	// Predict the result given the input
	template <int Batch>
	mat<Scalar, Outputs, Batch> predict(
		const mat<Scalar, Inputs, Batch> &input)
	{
		if constexpr (sizeof...(HiddenLayers) == 0)
		{
			// If no hidden layers, just return the first layer result
			return std::get<0>(layers).feedforward(input);
		}
		else
		{
			// Otherwise calculate the intermediate result
			// and then feed it through to the hidden layers
			auto output = std::get<0>(layers).feedforward(input);
			return Net<Scalar, Cost, HiddenLayers...>(
					   neural::pop_front(layers))
				.predict(output);
		}
	}

	// The main component of training
	template <int Batch>
	mat<Scalar, Inputs, Batch> feedforward_backward(
		Scalar learning_rate,
		mat<Scalar, Inputs, Batch> input,
		mat<Scalar, Outputs, Batch> expected)
	{
		if constexpr (sizeof...(HiddenLayers) == 0)
		{
			// If no hidden layers left
			// get the first layer to feedforward_backward
			// using the cost layer as it's next step
			return std::get<0>(layers)
				.feedforward_backward(
					learning_rate, input, expected,
					CostLayer<Scalar, Cost, Outputs>());
		}
		else
		{
			// Otherwise, feedforward_backward with the
			// remaining hiddenlayers as the next step
			return std::get<0>(layers)
				.feedforward_backward(
					learning_rate, input, expected,
					Net<Scalar, Cost, HiddenLayers...>(
						neural::pop_front(layers)));
		}
	}

private:
	template <int n>
	void random()
	{
		// Set all the layers to random
		std::get<n>(layers).random();
		if constexpr (sizeof...(HiddenLayers) != n)
			random<n + 1>();
	}

	template <int n>
	std::ostream &save(std::ostream &os) const
	{
		// Save all the layers
		os << std::get<n>(layers);
		if constexpr (n != sizeof...(HiddenLayers))
		{
			save<n + 1>(os);
		}
		return os;
	}

	template <int n>
	std::istream &load(std::istream &is)
	{
		// Load all the layers
		is >> std::get<n>(layers);
		if constexpr (n != sizeof...(HiddenLayers))
		{
			load<n + 1>(is);
		}
		return is;
	}
};

template <
	typename S,
	typename C,
	typename IL, typename... HLs>
const decltype(utils::__get_info<IL, HLs...>()) Net<S, C, IL, HLs...>::info
	 = utils::__get_info<IL, HLs...>();

} // namespace neural