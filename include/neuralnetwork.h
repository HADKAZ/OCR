#pragma once
#include "stdlib.h"


typedef struct
{
    double activation;
    double *weights;
    double bias;

    /*
     * The following values will be used
     * to save the delta during the backpropagation
     */
    double delta;
    double error;
    double *wdelta;
    double bdelta;
} Neuron;

typedef struct
{
    int size;
    Neuron *neurons;
} Layer;

typedef struct
{
    Layer* layers;
    int layerCount;

    double (*cost)(double, double);

    void (*hidden_activation)(Layer*);
    void (*hidden_activation_derivative)(Layer*);

    void (*output_activation)(Layer*);
    void (*output_activation_derivative)(Layer*, double(*)(double, double), const double*);

} NeuralNetwork;

NeuralNetwork initNetwork(int layerCount, int* layerSizes, char* hidden_activation, char* output_activation);
double* execute(NeuralNetwork net, double* input);
void training(NeuralNetwork net,
              double*** dataSet,
              int dataSize,
              double learningRate,
              int epoch,
              int batchsize
);

// Utils
void printNeuralNetworkShape(NeuralNetwork net);
void freeNetwork(NeuralNetwork net);

// Save & load

void saveNeuralNetwork(NeuralNetwork net, const char *filePath);
NeuralNetwork loadNeuralNetwork(const char *filePath);