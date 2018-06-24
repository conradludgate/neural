#pragma once

#include <Eigen/Dense>
#include <tuple>
#include <iostream>
#include <functional>

#include "neural/util.hpp"

class ANN;

namespace neural
{

template<int A, int B>
class Layer //: public ANN
{
public:

	static const int I = A;
	static const int O = B;

	void random()
	{
		weight = mat<B, A>::Random();
		bias = vec<B>::Random();
	}

	std::ostream& save(std::ostream& os)
	{
		for (int i = 0; i < B; i++)
		{
			for (int j = 0; j < A; j++)
			{
				os << weight[i, j];
			}
			os << bias[i];
		}

		return os;
	}

	std::istream& load(std::istream& is)
	{
		for (int i = 0; i < B; i++)
		{
			for (int j = 0; j < A; j++)
			{
				is << weight[i, j];
			}
			is << bias[i];
		}

		return is;
	}

	vec<B> feedforward(vec<A> input)
	{
		last_input = input;
		last_output = g(weight * input + bias);

		return last_output;
	}

	// error = δError/δOutput
	vec<A> feedbackward(vec<B> error)
	{
		
		error = gprime(error, last_output);

		vec<A> dinput = weight.transpose() * error;
		bias += error;
		weight += error * last_input.transpose();

		return dinput;
	}

private:
	vec<A> last_input;
	vec<B> last_output;

	mat<B, A> weight;
	vec<B> bias;

	vec<B> g(const vec<B>& v)
	{
		return v.unaryExpr(std::ref(Layer<A, B>::sigmoid));
		//return v.unaryExpr(std::ref(Layer<A, B>::relu));
	}

	// error = δError/δOutput * δOutput/δg
	vec<B> gprime(const vec<B>& error, const vec<B>& output)
	{	
		return error.binaryExpr(output, std::ref(Layer<A, B>::sigmoid_prime));
		//return error.binaryExpr(output, std::ref(Layer<A, B>::relu_prime));
	}

	static Scalar sigmoid(const Scalar& x)
	{
		return 1 / (exp(-x) + 1);
	}

	static Scalar sigmoid_prime(const Scalar& a, const Scalar& b)
	{
		return a * (1 - b) * b;
	}

	static Scalar relu(const Scalar& x)
	{
		if (x > 0) 
			return x;
		else
			return 0;
	}

	static Scalar relu_prime(const Scalar& a, const Scalar& b)
	{
		if (b > 0)
			return a;
		else
			return 0;
	}
};

}