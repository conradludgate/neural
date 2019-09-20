#pragma once

#include <iostream>
#include <type_traits>

#include "neural/util.hpp"
#include "neural/cost_layer.hpp"

namespace neural
{

// template<int I, int O>
// class ANN
// {
// public:
// 	virtual void random() = 0;
// 	virtual vec<Outputs>& feedforward(vec<Inputs>&) = 0;
// 	virtual vec<Inputs>& feedbackward(vec<Outputs>&) = 0;
// 	virtual std::ostream& operator<<(std::ostream&) = 0;
// 	virtual std::istream& operator>>(std::istream&) = 0;
// };

template <typename Scalar, typename Cost, typename InputLayer, typename... HiddenLayers>
class Net //: public ANN
{
private:
	std::tuple<InputLayer, HiddenLayers...> layers;

public:
	Net() {}
	Net(std::tuple<InputLayer, HiddenLayers...> layers) : layers(layers) {}

	static const int Inputs = InputLayer::Inputs;
	static const int Outputs = std::tuple_element<sizeof...(HiddenLayers), decltype(layers)>::type::Outputs;

	void random() { random<0>(); }
	std::ostream &operator<<(std::ostream &os) { return save<0>(os); }
	std::istream &operator>>(std::istream &is) { return load<0>(is); }

	template <int Batch>
	void train(Scalar lr, mat<Scalar, Inputs, Batch> input, mat<Scalar, Outputs, Batch> expected)
	{
		feedforward_backward(lr, input, expected);
	}

	template <int Batch>
	mat<Scalar, Outputs, Batch> predict(const mat<Scalar, Inputs, Batch> &input)
	{
		return feedforward<0, Inputs, Batch>(input);
	}

	vec<Scalar, Outputs> predict(const vec<Scalar, Inputs> &input)
	{
		return feedforward<0, Inputs, 1>(input);
	}

	template <int Batch>
	mat<Scalar, Inputs, Batch> feedforward_backward(Scalar learning_rate, mat<Scalar, Inputs, Batch> input, mat<Scalar, Outputs, Batch> expected)
	{
		if constexpr (sizeof...(HiddenLayers) == 0)
		{
			return std::get<0>(layers)
				.feedforward_backward(
					learning_rate, input, expected, CostLayer<Scalar, Cost, Outputs>());
		}
		else
		{
			return std::get<0>(layers)
				.feedforward_backward(
					learning_rate, input, expected,
					Net<Scalar, Cost, HiddenLayers...>(neural::pop_front(layers)));
		}
	}

private:
	template <int n>
	void random()
	{
		std::get<n>(layers).random();
		if constexpr (sizeof...(HiddenLayers) != n)
			random<n + 1>();
	}

	template <int n>
	std::ostream &save(std::ostream &os)
	{
		std::get<n>(layers).save(os);
		if constexpr (n != sizeof...(HiddenLayers))
		{
			save<n + 1>(os);
		}
		return os;
	}

	template <int n>
	std::istream &load(std::istream &is)
	{
		std::get<n>(layers).load(is);
		if constexpr (n != sizeof...(HiddenLayers))
		{
			load<n + 1>(is);
		}
		return is;
	}

	template <int n, int inputs, int Batch>
	mat<Scalar, Outputs, Batch> feedforward(mat<Scalar, inputs, Batch> input)
	{
		auto &l = std::get<n>(layers);
		if constexpr (n == sizeof...(HiddenLayers))
		{
			return l.feedforward(input);
		}
		else
		{
			return feedforward<n + 1,
							   std::tuple_element<n, decltype(layers)>::type::Outputs>(l.feedforward(input));
		}
	}
};

} // namespace neural