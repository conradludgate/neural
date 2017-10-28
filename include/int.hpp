#pragma once

#include <vector>
#include <iterator>
#include <cstdlib>

#include "neural/train.hpp"
#include "neural/net.hpp"

class IntIter : public std::iterator<std::input_iterator_tag, int>
{
public:
	IntIter(int* v) : p(v) {}
	IntIter() : end(true) {}

	IntIter operator ++()
	{
		end = true;

		return *this;
	}

	bool operator !=(const IntIter& rhs) const {return end!=rhs.end;}

	int& operator *() {return *p;}

private:
	int* p;
	bool end = false;
};

class Int 
{
public:
	Int(int value = 0) : m_value(value) {}

	int getValue() 
	{
		return m_value;
	}

	void setValue(int value)
	{
		m_value = value;
	}

	Int operator =(std::vector<float> value)
	{
		m_value = value[0];
	}

	operator float()
	{
		return m_value;
	}

	IntIter begin() { return IntIter(&m_value); }
	IntIter end() { return IntIter(); }
private:
	int m_value;
};

class AddTrainer: public neural::Trainer
{
public:
	AddTrainer()
	{
		setSize({2, 1});
	}

	float costf(std::vector<float> output, std::vector<float> expected)
	{
		return output[0] - expected[0];
	}

	std::vector<neural::TestData> data_set()
	{
		std::vector<neural::TestData> test_data(10000);

		for (auto i = test_data.begin(); i != test_data.end(); ++i)
		{
			float a = rand() % 1024;
			float b = rand() % 1024;

			(*i).input = {a, b};
			(*i).expected = {a + b};
		}

		return test_data;
	}
};