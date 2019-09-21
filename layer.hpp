#pragma once

#include <Eigen/Dense>
#include <tuple>
#include <iostream>

#include "neural/util.hpp"

namespace neural
{

template <typename Scalar, int I, int O, typename Activation>
class LinearLayer
{
public:
	static const int Inputs = I;
	static const int Outputs = O;

	void random()
	{
		weight = mat<Scalar, Outputs, Inputs>::Random();
		bias = vec<Scalar, Outputs>::Random();
	}

	std::ostream &save(std::ostream &os)
	{
		os.write((char *)weight.data(), sizeof(Scalar) * Inputs * Outputs);
		os.write((char *)bias.data(), sizeof(Scalar) * Outputs);

		return os;
	}

	std::istream &load(std::istream &is)
	{
		is.read((char *)weight.data(), sizeof(Scalar) * Inputs * Outputs);
		is.read((char *)bias.data(), sizeof(Scalar) * Outputs);

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

} // namespace neural