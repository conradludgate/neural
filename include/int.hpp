#pragma once

#include <vector>
#include <iterator>

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