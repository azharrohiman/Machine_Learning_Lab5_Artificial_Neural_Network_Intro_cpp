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
	//cout << &neural1 << endl;

	// weights between input and hidden nodes
	MatrixXd weights_input_hidden(2, 3);

	weights_input_hidden(0, 0) = 0.1;
	weights_input_hidden(0, 1) = 0.2;
	weights_input_hidden(0, 2) = 0.5;

	weights_input_hidden(1, 0) = -0.4;
	weights_input_hidden(1, 1) = 1.0;
	weights_input_hidden(1, 2) = -0.6;

	cout << "Weights between input and hidden nodes are: " << endl << weights_input_hidden << endl << endl;

	// input values for the input nodes
	MatrixXd input(3, 1);

	input(0, 0) = 1.3;
	input(1, 0) = 2.7;
	input(2, 0) = 0.8;

	cout << "Input values are: " << endl << input << endl << endl;

	// weights between hidden and output nodes
	MatrixXd weights_hidden_output(1, 2);

	weights_hidden_output(0, 0) = 0.8;
	weights_hidden_output(0, 1) = 1.0;

	cout << "Weights between hidden and output nodes are: " << endl << weights_hidden_output << endl << endl;

	// biases
	MatrixXd hidden_biases(2, 1);

	hidden_biases(0, 0) = 0.1;
	hidden_biases(1, 0) = -0.3;

	cout << "Biases for the hidden nodes are: " << endl << hidden_biases << endl << endl;

	MatrixXd output_bias(1, 1);

	output_bias(0, 0) = -0.3;

	cout << "Bias for the output node is: " << endl << output_bias << endl << endl;

	return 0;
}
