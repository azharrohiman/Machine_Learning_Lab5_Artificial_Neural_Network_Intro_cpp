#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <sstream>

#include "NeuralNetwork.h"

using namespace std;

namespace RHMMUH005 
{
	NeuralNetwork::NeuralNetwork() {

	}

	NeuralNetwork::NeuralNetwork(int input, int hidden, int output, MatrixXd input_hidden) {
		num_input_nodes = input;
		num_hidden_nodes = hidden;
		num_output_nodes = output;

		weights_input_hidden = input_hidden;
	}
}
