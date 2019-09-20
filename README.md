# Neural Network

**Warning: This is a WIP and an experiment. Do not use for production**

### Dependencies
*	eigen - C++ Linear Algebra Library

## Usage
See example/mnist for example usage.

### Concepts

The main concept of this lib is based around "neuron layers". 

Here's an example of how to make a neural network type

```c++
typedef neural::Net<
	float,                                                                // Scalar type
	neural::cost::MSE,                                                    // Cost function
	neural::LinearLayer<float, IMAGE_SIZE, 30, neural::activation::Relu>, // Input Layer
	neural::LinearLayer<float, 30, 10, neural::activation::Sigmoid>>      // Output Layer
	NN;
```

The first template parameter is the `Scalar` type. Typically a float or a double, but can be whatever as long as it supports * and +

The second parameter is the `Cost` type, which is a class that contains one static function called `cost` that takes in the output and the expected output and calculates the error

After that, you can add 1 or more `Layer`s. Here we see 2 `LinearLayer`s, which take in the Scalar value, the input size, the output size and the `Activation` type