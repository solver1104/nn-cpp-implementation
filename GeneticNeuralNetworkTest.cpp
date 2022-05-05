//#pragma warning( disable : 4996 )
#include <bits/stdc++.h>

#pragma GCC target ("avx2")
#pragma GCC optimization ("O3")
#pragma GCC optimization ("unroll-loops")

using namespace std;

int buffer;
const int popSize = 100;
int evolutionCycles = 398;
int evolutionReps = 1;
const int inputDimensions = 784;
const int hiddenLayer1Nodes = 300;
const int outputSize = 10;
float chanceEdgeEvolve = 0.001;
float softMaxSum;
const int trainingCaseNumber = 1990;
const int testCaseNumber = 10;
float leakyRLUConst = 0.001;
float totalAccuracy;
float trainingCases[trainingCaseNumber][inputDimensions];
float testCases[testCaseNumber][inputDimensions];
float edgesL1[popSize][inputDimensions][hiddenLayer1Nodes];
float edgesL2[popSize][hiddenLayer1Nodes][outputSize];
float offspringEdgesL1[popSize][inputDimensions][hiddenLayer1Nodes];
float offspringEdgesL2[popSize][hiddenLayer1Nodes][outputSize];
int trainingCasesAnswers[trainingCaseNumber];
float tempSum;
float tempRightWeight;
float tempWeightsL1[hiddenLayer1Nodes];
float tempWeightsL2[outputSize];
float accuracies[popSize];
int prediction;
float predictionConfidence;
float e = 2.7182818284;
float prefixes[popSize];

unsigned seed = chrono::system_clock::now().time_since_epoch().count();
default_random_engine generator(seed);
normal_distribution<double> distribution(0.0, chanceEdgeEvolve);
normal_distribution<double> initdistribution(0.0, 1.0);
uniform_real_distribution<double> regdistribution(0.0, 1.0);

int main()
{
	
	string problemName = "data";
	string outputName = "output";
	ifstream cin(problemName + ".txt");
	ofstream cout(outputName + ".txt");

	ios::sync_with_stdio(0);
	cin.tie(0);

	for (int i = 0; i < trainingCaseNumber; i++) {
		for (int j = 0; j < inputDimensions; j++) {
			cin >> trainingCases[i][j];
			trainingCases[i][j] /= 256;
		}
		cin >> trainingCasesAnswers[i];
	}

	for (int i = 0; i < testCaseNumber; i++) {
		for (int j = 0; j < inputDimensions; j++) {
			cin >> testCases[i][j];
			testCases[i][j] /= 256;
		}
		cin >> buffer;
		cout << buffer << "\n";
	}

	cout << "\n";
	
	for (int i = 0; i < popSize; i++) {
		for (int j = 0; j < inputDimensions; j++) {
			for (int k = 0; k < hiddenLayer1Nodes; k++) offspringEdgesL1[i][j][k] = initdistribution(generator);
		}

		for (int j = 0; j < hiddenLayer1Nodes; j++) {
			for (int k = 0; k < outputSize; k++) offspringEdgesL2[i][j][k] = initdistribution(generator);
		}
	}

	for (int i = 0; i < evolutionCycles; i++) {
		for (int j = 0; j < popSize; j++) {
			fill(accuracies, accuracies + popSize, 0);
			for (int k = 0; k < inputDimensions; k++) {
				for (int l = 0; l < hiddenLayer1Nodes; l++) edgesL1[j][k][l] = offspringEdgesL1[j][k][l] + distribution(generator);
			}
			for (int k = 0; k < hiddenLayer1Nodes; k++) {
				for (int l = 0; l < outputSize; l++) edgesL2[j][k][l] = offspringEdgesL2[j][k][l] + distribution(generator);
			}
			for (int k = 0; k < (evolutionReps * trainingCaseNumber) / evolutionCycles; k++) {
				tempSum = 0.0;
				softMaxSum = 0.0;
				fill(tempWeightsL1, tempWeightsL1 + hiddenLayer1Nodes, 0);
				fill(tempWeightsL2, tempWeightsL2 + outputSize, 0);
				for (int l = 0; l < inputDimensions; l++) {
					for (int m = 0; m < hiddenLayer1Nodes; m++) {
						tempWeightsL1[m] += ((edgesL1[j][l][m] * trainingCases[i * (trainingCaseNumber / evolutionCycles) + (k % (trainingCaseNumber / evolutionCycles))][l] < 0) ? leakyRLUConst * edgesL1[j][l][m] * trainingCases[i * (trainingCaseNumber / evolutionCycles) + (k % (trainingCaseNumber / evolutionCycles))][l] : edgesL1[j][l][m] * trainingCases[i * (trainingCaseNumber / evolutionCycles) + (k % (trainingCaseNumber / evolutionCycles))][l]);
					}
				}
				for (int l = 0; l < hiddenLayer1Nodes; l++) {
					for (int m = 0; m < outputSize; m ++) softMaxSum += edgesL2[j][l][m] * pow(e, tempWeightsL1[l]);
				}
				for (int l = 0; l < hiddenLayer1Nodes; l++) {
					for (int m = 0; m < outputSize; m++) {
						tempWeightsL2[m] += ((edgesL2[j][l][m] * pow(e, tempWeightsL1[l])) / softMaxSum);
					}
				}

				tempRightWeight = tempWeightsL2[trainingCasesAnswers[i * (trainingCaseNumber/evolutionCycles) + (k % (trainingCaseNumber / evolutionCycles))]];
				for (int l = 0; l < outputSize; l++) tempSum += tempWeightsL2[l];
				accuracies[j] += (tempRightWeight / tempSum) * (tempRightWeight / tempSum);
			}
		}

		totalAccuracy = 0.0;
		for (int j = 0; j < popSize/10; j++) totalAccuracy += accuracies[j];

		prefixes[0] = accuracies[0] / totalAccuracy;
		for (int j = 1; j < popSize/10; j++) {
			prefixes[j] = prefixes[j - 1] + (accuracies[j] / totalAccuracy);
		}

		for (int j = 0; j < popSize; j++) {
			for (int k = 0; k < inputDimensions; k++) {
				for (int l = 0; l < hiddenLayer1Nodes; l++) {
					offspringEdgesL1[j][k][l] = edgesL1[lower_bound(prefixes, prefixes + popSize, regdistribution(generator)) - prefixes][k][l];
				}
			}

			for (int k = 0; k < hiddenLayer1Nodes; k++) {
				for (int l = 0; l < outputSize; l++) {
					offspringEdgesL2[j][k][l] = edgesL2[lower_bound(prefixes, prefixes + popSize, regdistribution(generator)) - prefixes][k][l];
				}
			}
		}
	}
	
	for (int k = 0; k < testCaseNumber; k++) {
		tempSum = 0.0;
		softMaxSum = 0.0;
		tempRightWeight = 0.0;
		fill(tempWeightsL1, tempWeightsL1 + hiddenLayer1Nodes, 0);
		fill(tempWeightsL2, tempWeightsL2 + outputSize, 0);
		for (int l = 0; l < inputDimensions; l++) {
			for (int m = 0; m < hiddenLayer1Nodes; m++) {
				tempWeightsL1[m] += ((offspringEdgesL1[0][l][m] * testCases[k][l] < 0) ? leakyRLUConst * offspringEdgesL1[0][l][m] * testCases[k][l] : offspringEdgesL1[0][l][m] * testCases[k][l]);
			}
		}
		for (int l = 0; l < hiddenLayer1Nodes; l++) {
			for (int m = 0; m < outputSize; m++) softMaxSum += offspringEdgesL2[0][l][m] * pow(e, tempWeightsL1[l]);
		}
		for (int l = 0; l < hiddenLayer1Nodes; l++) {
			for (int m = 0; m < outputSize; m++) {
				tempWeightsL2[m] += ((offspringEdgesL2[0][l][m] * pow(e, tempWeightsL1[l])) / softMaxSum);
			}
		}

		for (int l = 0; l < outputSize; l++) {
			if (tempRightWeight < tempWeightsL2[l]) {
				tempRightWeight = tempWeightsL2[l];
				prediction = l;
			}
		}
		for (int l = 0; l < outputSize; l++) tempSum += tempWeightsL2[l];
		predictionConfidence = (tempRightWeight / tempSum) * (tempRightWeight / tempSum);

		cout << prediction << " " << predictionConfidence << "\n";
	}
	
	for (int i = 0; i < inputDimensions; i++) {
		for (int j = 0; j < hiddenLayer1Nodes; j++) {
			cout << offspringEdgesL1[0][i][j] << "\n";
		}
	}
	
	for (int i = 0; i < hiddenLayer1Nodes; i++) {
		for (int j = 0; j < outputSize; j++) {
			cout << offspringEdgesL2[0][i][j] << "\n";
		}
	}
}