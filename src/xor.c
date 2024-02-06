//
// Created by martin on 03/11/23.
//
#include "neuralnetwork.h"
#include <stdio.h>
#include <math.h>
#define inputSize 2
#define outputSize 1
#define dataSetSize 4

// Possible inputs for the xor function
double inputs[dataSetSize][inputSize] =
        {
                {0,0},
                {0,1},
                {1,0},
                {1,1}
        };

// Outputs of the xor function that match the input
double outputs[dataSetSize][outputSize] =
        {
                {0.0},
                {1.0},
                {1.0},
                {0.0}
        };

double ***dataSetGenerator(
        double input[][inputSize],
        double output[][outputSize],
        int size
        )
{
    /*
     * Generates a dataSet from the given input and output
     */
    double ***dataSet = malloc(size*sizeof(double**));
    for (int i = 0; i < size; i++) {
        dataSet[i] = malloc(2 * sizeof(double*));
        dataSet[i][0] = malloc(inputSize*sizeof(double));
        for(int j = 0; j < inputSize; j++) {
            dataSet[i][0][j] = input[i][j];
        }
        dataSet[i][1] = malloc(outputSize*sizeof(double));
        for(int j = 0; j < outputSize; j++) {
            dataSet[i][1][j] = output[i][j];
        }
    }

    return dataSet;
}

void freeDataSet(double*** dataSet, int size)
{
    /*
     * Properly free the given dataSet
     * the original inputs and outputs are not freed
     */
    for(int i = 0; i < size; i++)
    {
        free(dataSet[i][0]);
        free(dataSet[i][1]);
        free(dataSet[i]);
    }
    free(dataSet);
}

int main(){
    // Define values
    double ***dataSet = dataSetGenerator(inputs, outputs, dataSetSize);
    int layerSizes[] = {2, 2, 1};
    double learningRate = 0.5;
    int epoch = 20000;

    printf("---  Settings  ---\n");
    printf("learning_rate -> %f\n", learningRate);
    printf("epoch -> %i\n", epoch);
    // Create network
    printf("\n--- Initialisation ---\n");
    printf("Generating neural network...\n");

    NeuralNetwork xor = initNetwork(3, layerSizes, "relu", "sigmoid");

    printf("Neural network generated!\n");
    printNeuralNetworkShape(xor);

    printf("\nTraining neural network...\n");
    training(xor, dataSet, dataSetSize, learningRate, epoch, 32);

    printf("Neural network trained!\n");

    printf("\n---   Tests    ---\n");
    // Test the network
    for (int i = 0; i < dataSetSize; i++)
    {
        double *input = dataSet[i][0];
        double *expectedOutput = dataSet[i][1];

        double *output = execute(xor, input);

        // Print input
        printf("[ ");
        for(int j = 0; j < inputSize; j++)
            printf("%i ", (int)round(input[j]));

        printf("] -> %i (%i)\n", (int)round(output[0]), (int)round(expectedOutput[0]));
        free(output);
    }

    freeDataSet(dataSet, dataSetSize);
    freeNetwork(xor);
}