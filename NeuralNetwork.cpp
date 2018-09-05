#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <math.h>

#include "NeuralNetwork.h"

using namespace std;

namespace RHMMUH005 
{
	NeuralNetwork::NeuralNetwork() {

	}

	NeuralNetwork::NeuralNetwork(int input, int hidden, int output, MatrixXd input_values, MatrixXd input_hidden, MatrixXd hidden_output, MatrixXd biases_h, MatrixXd bias_output) {
		num_input_nodes = input;
		num_hidden_nodes = hidden;
		num_output_nodes = output;

		inputs = input_values;

		weights_input_hidden = input_hidden;
		weights_hidden_output = hidden_output;

		hidden_biases = biases_h;
		output_bias = bias_output;
	}

	void NeuralNetwork::feedForward() {
		MatrixXd hidden_mat = weights_input_hidden * inputs;
		cout << "Multiplication of weights between input and hidden nodes and input values is: " << endl;
		cout << hidden_mat << endl;

		hidden_mat = hidden_mat + hidden_biases;
		cout << "Addition of hidden biases is: " << endl;
		cout << hidden_mat << endl;

		MatrixXd hidden(2, 1);
		hidden(0, 0) = sigmoid(hidden_mat(0, 0));
		hidden(1, 0) = sigmoid(hidden_mat(1, 0));

		cout << "Value after sigmoid activation function of first neuron is: " << hidden(0, 0) << endl;
		cout << "Value after sigmoid activation function of second neuron is: " << hidden(1, 0) << endl;

		MatrixXd output_mat = weights_hidden_output * hidden;
		cout << "Multiplication of weights between hidden and output nodes and hidden values is: " << endl;
		cout << output_mat << endl;

		output_mat = output_mat + output_bias;
		cout << "Addition of output bias is: " << output_mat << endl;

		cout << "Value after sigmoid activation function of output node is: " << sigmoid(output_mat(0, 0)) << endl;
	}

	double NeuralNetwork::sigmoid(double val) {
		double exp_value;
     		double return_value;

     		/*** Exponential calculation ***/
     		exp_value = exp((double) -val);

     		/*** Final sigmoid value ***/
     		return_value = 1 / (1 + exp_value);

     		return return_value;
	}
}
