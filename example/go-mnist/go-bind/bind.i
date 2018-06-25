%module NN

%{
#include "bind.h"	
%}

%rename(Wrapped_NN) NN;
%rename(Wrapped_Train) NN::Train(float lr, float input[784], float target[10]);
%rename(Wrapped_Predict) NN::Predict(float input[784], float output[10]);

class NN
{
public:
	NN();
	void Train(float lr, float input[784], float target[10]);
	void Predict(float input[784], float output[10]);
};

%insert(cgo_comment_typedefs) %{
#cgo CXXFLAGS: -std=c++17 -I ../../../.. -I /usr/include/eigen3	
%}

%insert(go_wrapper) %{

type NN interface {
	Wrapped_NN
	Train(lr float32, input [784]float32, target [10]float32)
	Predict(input [784]float32) [10]float32
}

func NewNN() NN {
	return NewWrapped_NN().(SwigcptrWrapped_NN)
}

func (nn SwigcptrWrapped_NN) Train(lr float32, input [784]float32, target [10]float32) {
	nn.Wrapped_Train(lr, &input[0], &target[0])
}

func (nn SwigcptrWrapped_NN) Predict(input [784]float32) (output [10]float32) {
	nn.Wrapped_Predict(&input[0], &output[0])
	return output
}

%}

%include "bind.h"