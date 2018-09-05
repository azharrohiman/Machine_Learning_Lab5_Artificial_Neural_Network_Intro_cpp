#include <string>
#include <iostream>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

#include "NeuralNetwork.h"

using namespace Eigen;
using Eigen::MatrixXd;
using namespace std;

int main(int argc, char* argv[]) {

	//RHMMUH005::NeuralNetwork neural1;

	// weights between input and hidden nodes
	MatrixXd weights_input_hidden(2, 3);

	weights_input_hidden(0, 0) = 0.1;
	weights_input_hidden(0, 1) = 0.2;
	weights_input_hidden(0, 2) = 0.5;

	weights_input_hidden(1, 0) = -0.4;
	weights_input_hidden(1, 1) = 1.0;
	weights_input_hidden(1, 2) = -0.6;

	// input values for the input nodes
	MatrixXd input(3, 1);

	input(0, 0) = 1.3;
	input(1, 0) = 2.7;
	input(2, 0) = 0.8;

	// weights between hidden and output nodes
	MatrixXd weights_hidden_output(1, 2);

	weights_hidden_output(0, 0) = 0.8;
	weights_hidden_output(0, 1) = 1.0;

	// biases
	MatrixXd hidden_biases(2, 1);

	hidden_biases(0, 0) = 0.1;
	hidden_biases(1, 0) = -0.3;

	MatrixXd output_bias(1, 1);

	output_bias(0, 0) = -0.3;

	RHMMUH005::NeuralNetwork neuralNetworkObj(3, 2, 1, input, weights_input_hidden, weights_hidden_output, hidden_biases, output_bias);
	neuralNetworkObj.feedForward();

	return 0;
}
