#pragma once

#include <vector>
#include <iterator>
#include <cstdlib>

#include "neural/train.hpp"

class AddTrainer: public neural::Trainer<2, 1>
{
public:
	AddTrainer()
	{
		this->Random();
	}

	neural::TestData<2, 1> dataf()
	{
		neural::TestData<2, 1> td;

		float a = rand() % 1024;
		float b = rand() % 1024;

		td.input = {a, b};
		td.expected.setConstant(a + b);

		return td;
	}

	virtual float costf(vec<1> output, vec<1> expected)
	{
		auto diff = expected - output;
		return diff.coeff(0);
	}
};