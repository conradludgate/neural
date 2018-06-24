#pragma once

template<int dim1, int... dims>
int tensor_size()
{
	if constexpr(sizeof...(dims) == 0)
	{
		return dim1;
	}
	return dim1 * tensor_size<dims...>()
}

int tensor_size() { return 0; }

template<typename Scalar, int... Dimensions>
class Tensor
{
private:
	Scalar values[tensor_size<Dimensions...>()];
}