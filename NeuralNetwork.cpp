#include "iostream"
#include "vector"
#include "stdlib.h"
#include "TNN.h"
#include "math.h"
#define COUNT_INPUT 4

int main(int argc, char* argv[]){
	TinyNeuralNetwork tnn(2, 4, 1, 0.1);
	std::vector< std::vector<double> > input;
	input.resize(COUNT_INPUT);
	for (int i = 0; i < COUNT_INPUT; ++i){
		input[i].resize(2);
	}
	std::vector< std::vector<double> > teacher;
	teacher.resize(COUNT_INPUT);
	for (int i = 0; i < COUNT_INPUT; ++i){
		teacher[i].resize(1);
	}

	input[0][0] = 0.0; input[0][1] = 0.0; teacher[0][0] = 0.0;
	input[1][0] = 0.0; input[1][1] = 1.0; teacher[1][0] = 1.0;
	input[2][0] = 1.0; input[2][1] = 0.0; teacher[2][0] = 1.0;
	input[3][0] = 1.0; input[3][1] = 1.0; teacher[3][0] = 0.0;

	for (int i = 0; i < 1000; ++i){
		for (int j = 0; j < COUNT_INPUT; ++j){
			tnn.ForwardPropagation(input[j]);
			tnn.BackPropagation(teacher[j]);
		}
	}
	for (int i = 0; i < COUNT_INPUT; ++i){
		std::cout<<input[i][0]<<','<<input[i][1]<<" : "<<teacher[i][0]<<std::endl;
		tnn.ForwardPropagation(input[i]);
		std::cout<<tnn.output_out[0]<<std::endl;
	}

	return 0;
}