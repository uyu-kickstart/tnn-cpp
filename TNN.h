#include "stdlib.h"
#include "math.h"
#include "vector"

inline double randDouble(){
	return (double)rand() / ((double)RAND_MAX);
}

inline double sigmoid(double x){
	return 1 / (1 + exp(-x));
}

inline double defferentialSigmoid(double x){
	return sigmoid(x) * (1 - sigmoid(x));
}
inline double dSig_already(double x){
	return x * (1 - x);
}

class TinyNeuralNetwork{
	/*
	 * in 		: 入力層
	 * hidden 	: 隠れ層
	 * out 		: 出力層
	 * 
	 * output 	: 出力
	 * weight 	: 重み
	 * bias 	: バイアス、閾値
	 * alpha 	: 重みの変化分の係数
	 * eta 		: 最急降下法の係数
	*/
	public:
		int count_unit_in;
		int count_unit_hidden;
		int count_unit_out;
		double alpha;
		std::vector< std::vector<double> > weight_in_to_hidden;		//weight_in_to_hidden[count_unit_in][count_unit_hidden]
		std::vector< std::vector<double> > weight_hidden_to_out; 

		std::vector<double> output_in;					//入力するのはここに入れる
		std::vector<double> output_hidden;
		std::vector<double> output_out;

		/*
		 * weight_in_to_hidden,weight_hidden_to_out,output_in,output_hiddenにはバイアスの分が入る
		*/


		TinyNeuralNetwork(int unit_in, int unit_hidden, int unit_out, double alp){
			count_unit_in = unit_in;			//0番目のニューロンの入力は常に1,結合荷重は-θ
			count_unit_hidden = unit_hidden;
			count_unit_out = unit_out;
			alpha = alp;

			int count_unit_in_weight = count_unit_in + 1;
			weight_in_to_hidden.resize(count_unit_in_weight);		//+1はバイアスの分
			for(int i = 0; i < count_unit_in_weight; ++i){
				weight_in_to_hidden[i].resize(count_unit_hidden);
			}
			int count_unit_hidden_weight = count_unit_hidden + 1;
			weight_hidden_to_out.resize(count_unit_hidden_weight);
			for(int i = 0; i < count_unit_hidden_weight; ++i){
				weight_hidden_to_out[i].resize(count_unit_out);
			}
			output_in.resize(count_unit_in);
			output_hidden.resize(count_unit_hidden);
			output_out.resize(count_unit_out);

			Initialize();
		}

		void Initialize (){
			int count_unit_in_weight = count_unit_in + 1;
			srand((unsigned)time(NULL));
			//重み、閾値の初期化。乱数を突っ込む
			for (int i = 0; i < count_unit_in_weight; ++i){
				for (int j = 0; j < count_unit_hidden; ++j){
					weight_in_to_hidden[i][j] = randDouble();
				}
			}
			int count_unit_hidden_weight = count_unit_hidden + 1;
			for (int j = 0; j < count_unit_hidden_weight; ++j){
				for (int k = 0; k < count_unit_out; ++k){
					weight_hidden_to_out[j][k] = randDouble();
				}
			}
		}

		void ForwardPropagation(std::vector<double> &input){
			double sum;
			output_in = input;
			//入力層->隠れ層の伝搬
			for (int j = 0; j < count_unit_hidden; ++j){
				sum = 0.0;
				for (int i = 0; i < count_unit_in; ++i){
					sum += weight_in_to_hidden[i][j] * output_in[i];
				}
				sum += weight_in_to_hidden[count_unit_in][j] * 1;
				output_hidden[j] = sigmoid(sum);		//名前によって関数のポインタをさせば？
			}
			//隠れ層->出力層の伝搬
			for (int k = 0; k < count_unit_out; ++k){
				sum = 0.0;
				for (int j = 0; j < count_unit_hidden; ++j){
					sum += weight_hidden_to_out[j][k] * output_hidden[j];
				}
				sum += weight_hidden_to_out[count_unit_hidden][k] * 1;
				output_out[k] = sigmoid(sum);
			}
		}
		void BackPropagation(std::vector<double> &teacher_signal){
			double sum = 0.0;

			std::vector<double> delta_output_hidden;
			delta_output_hidden.resize(count_unit_hidden);

			std::vector<double> delta_output_out;
			delta_output_out.resize(count_unit_out);
			
			for (int k = 0; k < count_unit_out; ++k){
				delta_output_out[k] = (teacher_signal[k] - output_out[k]) * dSig_already(output_out[k]);
			}

			for (int j = 0; j < count_unit_hidden; ++j){
				sum = 0.0;
				for (int k = 0; k < count_unit_out; ++k){
					sum += weight_hidden_to_out[j][k] * delta_output_out[k];
				}
				delta_output_hidden[j] = dSig_already(output_hidden[j]) * sum;

				for (int i = 0; i < count_unit_in; ++i){
					weight_in_to_hidden[i][j] += alpha * delta_output_hidden[j] * output_in[i];
				}
				weight_in_to_hidden[count_unit_in][j] += alpha * delta_output_hidden[j] * 1.0;
			}

			for (int k = 0; k < count_unit_out; ++k){
				for (int j = 0; j < count_unit_hidden; ++j){
					weight_hidden_to_out[j][k] += alpha * delta_output_out[k] * output_hidden[j];
				}
				weight_hidden_to_out[count_unit_hidden][k] += alpha * delta_output_out[k] * 1.0;
			}
		}
};