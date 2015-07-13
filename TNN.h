#include "stdlib.h"
#include "math.h"
#include "vector"

/* TODO:
 * ・ForwardPropagationの引数を"配列の先頭のポインタ？"、返り値を"結果の配列？"
 * ・BackPropagationの引数を??
 * ・各メソッドの引数、返り値を決める
 * ・どんなメソッドが必要か？
 * ・コメントの追加
 */

//0.0~1.0の間の値を返す関数
inline double rand_from0to1(){
	return (double)rand() / ((double)RAND_MAX);
}

inline double sigmoid(double x){
	return 1 / (1 + exp(-x));
}

//引数にはsigmoid(x)の返り値を入れる
inline double defferentialSigmoid(double y){
	return y * (1 - y);
}
/* 参考
 * 
 * y = σ(x) = 1 / (1 + exp(-x))
 * σ'(x) = σ(x) * (1 - σ(x)) = y * (1 - y)
*/

/*
 * アルゴリズムは<http://www-ailab.elcom.nitech.ac.jp/lecture/neuro/menu.html>のバックプロパゲーションを参照
 * 参考:	<https://github.com/uyu-kickstart/tiny-nn/blob/master/src/neuralnetwork.js>
 * 		<http://edof.hatenablog.com/entry/2015/05/11/161342>
 * 
*/

class TinyNeuralNetwork{
	/*
	 * X 		: 入力層
	 * H 		: 隠れ層
	 * O 		: 出力層
	 * 
	 * weight 	: 重み
	 * alpha 	: 学習係数
	 * 
	 * count_unit_ : 各層のunit数
	*/
	public:
		int count_X;
		int count_H;
		int count_O;
		double alp;
		std::vector< std::vector<double> > weight_X_to_H;
		std::vector< std::vector<double> > weight_H_to_O; 

		std::vector<double> X;					//入力するのはここに入れる
		std::vector<double> H;
		std::vector<double> O;

		/*
		 * weight_X_to_H,weight_H_to_O,X,Hにはバイアスの分が入る
		*/


		TinyNeuralNetwork(int unit_in, int unit_hidden, int unit_out, double alpha){
			count_X = unit_in;
			count_H = unit_hidden;
			count_O = unit_out;
			alp = alpha;

			int count_X_p1 = count_I + 1;			//weightの数は
			weight_X_to_H.resize(count_X_p1);
			for(int i = 0; i < count_X_p1; ++i){
				weight_X_to_H[i].resize(count_H);
			}
			int count_H_p1 = count_H + 1;
			weight_H_to_O.resize(count_H_p1);
			for(int i = 0; i < count_H_p1; ++i){
				weight_H_to_O[i].resize(count_O);
			}
			X.resize(count_X);
			H.resize(count_H);
			O.resize(count_O);

			Initialize();
		}

		void Initialize (){
			int count_X_p1 = count_X + 1;
			srand((unsigned)time(NULL));
			//重み、閾値の初期化。乱数を突っ込む
			for (int i = 0; i < count_X_p1; ++i){
				for (int j = 0; j < count_H; ++j){
					weight_X_to_H[i][j] = rand_from0to1();
				}
			}
			int count_H_p1 = count_H + 1;
			for (int j = 0; j < count_H_p1; ++j){
				for (int k = 0; k < count_O; ++k){
					weight_H_to_O[j][k] = rand_from0to1();
				}
			}
		}

		void ForwardPropagation(std::vector<double> &input){
			double sum;
			X = input;
			//入力層->隠れ層の伝搬
			for (int j = 0; j < count_H; ++j){
				sum = 0.0;
				for (int i = 0; i < count_X; ++i){
					sum += weight_X_to_H[i][j] * X[i];
				}
				sum += weight_X_to_H[count_X][j] * 1;
				H[j] = sigmoid(sum);		//名前によって関数のポインタをさせば？
			}
			//隠れ層->出力層の伝搬
			for (int k = 0; k < count_O; ++k){
				sum = 0.0;
				for (int j = 0; j < count_H; ++j){
					sum += weight_H_to_O[j][k] * H[j];
				}
				sum += weight_H_to_O[count_H][k] * 1;
				O[k] = sigmoid(sum);
			}
		}

		void BackPropagation(std::vector<double> &teacher_signal){
			double sum = 0.0;
			std::vector<double> delta_H;
			delta_H.resize(count_H);

			std::vector<double> delta_O;
			delta_O.resize(count_O);
			
			for (int k = 0; k < count_O; ++k){
				delta_O[k] = (teacher_signal[k] - O[k]) * defferentialSigmoid(O[k]);
			}

			for (int j = 0; j < count_H; ++j){
				sum = 0.0;
				for (int k = 0; k < count_O; ++k){
					sum += weight_H_to_O[j][k] * delta_O[k];
				}
				delta_H[j] = defferentialSigmoid(H[j]) * sum;

				for (int i = 0; i < count_X; ++i){
					weight_X_to_H[i][j] += alpha * delta_H[j] * X[i];
				}
				weight_X_to_H[count_X][j] += alpha * delta_H[j] * 1.0;
			}

			for (int k = 0; k < count_O; ++k){
				for (int j = 0; j < count_H; ++j){
					weight_H_to_O[j][k] += alpha * delta_O[k] * H[j];
				}
				weight_H_to_O[count_H][k] += alpha * delta_O[k] * 1.0;
			}
		}
};