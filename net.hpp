#pragma once

#include <neural/layer.hpp>

#include <fstream>

namespace neural
{

template<int Inputs, int Outputs, typename A, typename... Bs>
class Net
{
public:
	void Zero() { zero<0>(); }
	void Random() { random<0>(); }
	void Constant(Scalar c) { constant<0>(); }

	vec<Outputs> predict(vec<Inputs> input)
	{
		return predict<0, A, Bs...>(input);
	}

	void train(Scalar lr, vec<Inputs> input, vec<Outputs> target)
	{
		train<sizeof...(Bs), Outputs>(lr * (target - predict(input)));
	}

private:
	std::tuple<A, Bs...> neuron_layers;

	template<int n>
	void zero()
	{
		std::get<n>(this->neuron_layers).Zero();
		if constexpr(sizeof...(Bs) != n)
			zero<n+1>();
	}

	template<int n>
	void random()
	{
		std::get<n>(this->neuron_layers).Random();
		if constexpr(sizeof...(Bs) != n)
			random<n+1>();
	}

	template<int n>
	void constant(Scalar c)
	{
		std::get<n>(this->neuron_layers).Constant(c);
		if constexpr(sizeof...(Bs) != n)
			constant<n+1>();
	}

	template<int n, typename _A, typename... _Bs>
	vec<Outputs> predict(vec<_A::Inputs> input)
	{
		if constexpr(sizeof...(_Bs) == 0)
		{
			return std::get<n>(this->neuron_layers).feedforward(input);
		}
		else
		{
			return predict<n+1, _Bs...>(std::get<n>(this->neuron_layers).feedforward(input));
		}
	}

	template<int n, int outputs>
	void train(vec<outputs> error)
	{
		if constexpr(n == 0)
		{
			std::get<n>(this->neuron_layers).feedbackward(error);
		}
		else
		{
			auto layer = std::get<n>(this->neuron_layers);
			train<n-1, decltype(layer)::Inputs>(layer.feedbackward(error));
		}
	}
};

}