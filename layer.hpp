#pragma once

#include <Eigen/Dense>
#include <tuple>
#include <iostream>

#include "neural/util.hpp"
#include "neural/serial.hpp"

namespace neural
{

template <typename Scalar, int I, int O, typename Activation>
class LinearLayer
{
public:
	static const int Inputs = I;
	static const int Outputs = O;
	static const int Size = sizeof(Scalar) * Inputs * Outputs + sizeof(Scalar) * Outputs;

	static const serial::layer_info info;

	void random()
	{
		weight = mat<Scalar, Outputs, Inputs>::Random();
		bias = vec<Scalar, Outputs>::Random();
	}

	friend std::ostream &operator<<(std::ostream &os, const LinearLayer &layer)
	{
		os.write((char *)layer.weight.data(), sizeof(Scalar) * Inputs * Outputs);
		os.write((char *)layer.bias.data(), sizeof(Scalar) * Outputs);

		return os;
	}

	friend std::istream &operator>>(std::istream &is, LinearLayer &layer)
	{
		is.read((char *)layer.weight.data(), sizeof(Scalar) * Inputs * Outputs);
		is.read((char *)layer.bias.data(), sizeof(Scalar) * Outputs);

		return is;
	}

	template <int Batch>
	mat<Scalar, Outputs, Batch> feedforward(mat<Scalar, Inputs, Batch> input)
	{
		auto linear = (weight * input).colwise() + bias;
		return Activation::template g<Scalar, Outputs, Batch>(linear);
	}

	template <int Batch, typename Next>
	mat<Scalar, Inputs, Batch> feedforward_backward(
		Scalar learning_rate,
		mat<Scalar, Inputs, Batch> input,
		mat<Scalar, Next::Outputs, Batch> expected,
		Next next)
	{
		auto linear = (weight * input).colwise() + bias;
		auto output = Activation::template g<Scalar, Outputs, Batch>(linear);

		auto error = next.feedforward_backward(learning_rate, output, expected);
		auto gprime = Activation::template gprime<Scalar, Outputs, Batch>(linear, output);
		error.array() *= gprime.array() / Batch;

		auto dinput = weight.transpose() * error;
		bias -= error.rowwise().sum();
		weight -= error * input.transpose();

		return dinput;
	}

private:
	mat<Scalar, Outputs, Inputs> weight;
	vec<Scalar, Outputs> bias;
};

template <typename S, int I, int O, typename A>
const auto LinearLayer<S, I, O, A>::info =
	serial::layer_info(serial::layer_type::linear_layer, Inputs, Outputs);

} // namespace neural