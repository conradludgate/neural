#pragma once

#include <tuple>

#include "neural/util.hpp"

namespace neural
{

template<int A, int B>
class RLayer : public ANN
{
public:

	static const int I = A;
	static const int O = B;

	void random()
	{
		weight = mat<B, A+B>::Random();
		bias = vec<B>::Random();
	}

	vec<B>& feedforward(vec<A>& input)
	{
		last_input << input, last_output;
		last_output = g(weight * last_input + bias);

		return last_output;
	}

	// error = δError/δOutput
	vec<A>& feedbackward(vec<B>& error)
	{
		gprime(error, last_output);

		vec<A> dinput = weight.transpose() * error;
		bias += error;
		weight += error * last_input.transpose();

		return dinput.segment<A>(0);
	}

private:
	vec<A+B> last_input;
	vec<B> last_output;

	mat<B, A+B> weight;
	vec<B> bias;

	vec<B> g(vec<B>& v)
	{
		//return 1 / (exp(-v.array()) + 1);
		 return v.array().max(0);
	}

	vec<B> g(vec<B> v)
	{
		//return 1 / (exp(-v.array()) + 1);
		 return v.array().max(0);
	}

	// error = δError/δOutput * δOutput/δg
	void gprime(vec<B>& error, vec<B>& output)
	{
		//error.array() *= (-output.array() + 1) * output.array();
		for (int i = 0; i < B; i++)
		{
			if (output[i] <= 0)
			{
				error[i] = 0;
			}
			else
			{
				error[i] = 1;
			}
		}
	}
};

}