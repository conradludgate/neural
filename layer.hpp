#pragma once

#include <Eigen/Dense>
#include <tuple>

typedef float Scalar;

template<int A>
using vec = Eigen::Matrix<Scalar, A, 1>;

template<int A, int B>
using mat = Eigen::Matrix<Scalar, A, B>;

namespace neural
{

template<int A, int B>
class NeuronLayer
{
public:

	static const int Inputs = A;
	static const int Outputs = B;

	void Zero()
	{
		weight = mat<B, A>::Zero();
		bias = vec<B>::Zero();
	}

	void Random()
	{
		weight = mat<B, A>::Random();
		bias = vec<B>::Random();
	}

	void Constant(Scalar c)
	{
		weight = mat<B, A>::Constant(c);
		bias = vec<B>::Constant(c);
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
		gprime(error, last_output);

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

	vec<B> g(vec<B> v)
	{
		for (int i = 0; i < B; ++i)
			v[i] = 1 / (1 + exp(-v[i]));

		return v;
	}

	void gprime(vec<B>& s, vec<B> v)
	{
		for (int i = 0; i < B; ++i)
			s[i] *= (1 - v[i]) * v[i];
	}
};

}