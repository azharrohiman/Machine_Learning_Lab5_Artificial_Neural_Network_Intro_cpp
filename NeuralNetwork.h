#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <memory>
#include <string>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

using namespace Eigen;
using Eigen::MatrixXd;

namespace RHMMUH005 {
	
	class NeuralNetwork {

		private:
			int num_input_nodes;
			int num_hidden_nodes;
			int num_output_nodes;

			MatrixXd inputs;

			MatrixXd weights_input_hidden;
			MatrixXd weights_hidden_output;

			MatrixXd hidden_biases;
			MatrixXd output_bias;

		public:
			NeuralNetwork();
			NeuralNetwork(int, int, int, MatrixXd, MatrixXd, MatrixXd, MatrixXd, MatrixXd);

			void feedForward();

			double sigmoid(double);

			double mse(double);

	};
}

#endif
