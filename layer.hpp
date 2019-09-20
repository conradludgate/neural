#pragma once

#include <Eigen/Dense>
#include <tuple>
#include <iostream>

#include "neural/util.hpp"

class ANN;

namespace neural
{

template <typename Scalar, int I, int O, typename Activation>
class LinearLayer //: public ANN
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
		for (int i = 0; i < Outputs; i++)
		{
			for (int j = 0; j < Inputs; j++)
			{
				os << weight[i, j];
			}
			os << bias[i];
		}

		return os;
	}

	std::istream &load(std::istream &is)
	{
		for (int i = 0; i < Outputs; i++)
		{
			for (int j = 0; j < Inputs; j++)
			{
				is << weight[i, j];
			}
			is << bias[i];
		}

		return is;
	}

	template <int Batch>
	mat<Scalar, Outputs, Batch> feedforward(mat<Scalar, Inputs, Batch> input)
	{
		return Activation::template g<Scalar, Outputs, Batch>(weight * input + bias);
	}

	template <int Batch, typename Next>
	mat<Scalar, Inputs, Batch> feedforward_backward(Scalar learning_rate, mat<Scalar, Inputs, Batch> input, mat<Scalar, Next::Outputs, Batch> expected, Next next)
	{
		auto output = Activation::template g<Scalar, Outputs, Batch>(weight * input + bias);

		auto error = Activation::template gprime<Scalar, Outputs>(
			next.feedforward_backward(learning_rate, output, expected),
			output);

		auto dinput = weight.transpose() * error;
		bias += error;
		weight += error * input.transpose();

		return learning_rate * dinput;
	}

private:
	vec<Scalar, Inputs> last_input;
	vec<Scalar, Outputs> last_output;

	mat<Scalar, Outputs, Inputs> weight;
	vec<Scalar, Outputs> bias;
};

} // namespace neural