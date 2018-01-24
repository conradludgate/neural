# Neural Network

**Warning: This is a WIP and an experiment. Do not use for production**

### Dependencies
*	boost/archive
*	eigen - C++ Linear Algebra Library

## Usage

To Train a network, 
you need to create a `neural::Trainer` class with a data function.

Let's say we're making a network with the amount of nodes being 16, 20, 20, 16
respectively, this is what our trainer would look like

```c++
class ExampleTrainer: public neural::Trainer<16, 20, 20, 8>
{
public:
	ExampleTrainer()
	{
		// Initialise the values
		this->Zero();
		// this->Random();
	}

	neural::TestData<16, 8> dataf()
	{
		neural::TestData<16, 8> td;

		// Set the values of td.input and td.expected

		return td;
	}
};
```

`td.input` is of type `Eigen::Matrix<float, 16, 1>` and `td.expected` is an `Eigen::Matrix<float, 8, 1>`.

Then to use the train the network, we do the following

```c++
func main()
{
	ExampleTrainer network;

	// Smaller = better resulting network
	// Larger = training will finish faster
	float tolerance = 0.01;

	// Too large and the network might not converge
	// Too small and the network will take ages to converge.
	float precision = 0.01;

	// Begin training, can take a very long time
	network.train(tolerance, precision);

	// Save the network to file
	std::ofstream ofs("network-data");
	boost::archive::binary_oarchive oa(ofs);
    oa << network;
}
```

Then to use it in a normal use environment

```c++
func main()
{
	neural::Net<16, 20, 20, 8> network;

	// Load the network from file
	std::ifstream ifs("network-data");
	boost::archive::binary_iarchive ia(ifs);
	ia >> network;

	// Use the network
	Eigen::Matrix<float, 16, 1> input;
	input.setRandom();

	std::cout << network.process(input) << std::endl;
}
```