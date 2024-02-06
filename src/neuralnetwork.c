#include "neuralnetwork.h"
#include <err.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
/*
 * Utilities functions
 */

double randD()
{
    /*
     * Return a double between 0 and 1
     */
    return (double)(((double)rand())/(double)RAND_MAX);
}

double normalRand()
{
    return (randD()+randD()+randD()+randD()-2.0)*1.724;
}

void shuffle(double ***array, int n) {
    if (n <= 1) return; // No need to shuffle if the array has 0 or 1 elements

    for (int i = n - 1; i > 0; i--) {
        // Generate a random index in the range [0, i]
        int j = rand() % (i + 1);

        // Swap array[i] and array[j]
        double **temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

void printNeuralNetworkShape(NeuralNetwork net)
{
    /*
     * Print the shape of the give NeuralNetwork
     */
    printf("Neural Network Shape:\n");
    for (int i = 0; i < net.layerCount; i++) {
        printf("Layer %i: %i neurons\n", i, net.layers[i].size);
    }
}

void printDATA(const double** input)
{
    /*
     * This function is meant to print the images in the MNIST dataSet
     */
    printf("TRAINING ON >\n");
    for(int i = 0; i < 28; i++)
    {
        for(int j = 0; j < 28; j++)
        {
            char val =' ';
            if (input[0][j * 28 + i] > 0.3)
                val = '?';
            if (input[0][j * 28 + i] > 0.5)
                val = '@';
            if (input[0][j * 28 + i] > 0.8)
                val = '#';
            printf("%c ",val);
        }
        printf("\n");
    }
    int i = 0;
    while(!input[1][i]){i++;}
    printf("Expected -> %0i\n", i);
}

/*
 * Activation functions
 */

double MSE(double got, double expected)
{
    /*
     * Mean squared error cost function
     */
    return got - expected;
}

double tanh_g (double z)
{
    return 1.0 - (z*z);
}

double sSigmoid(double z)
{
    return 1 / (1 + exp(-z));
}

void sigmoid(Layer *l)
{
    /*
     * Applies the sigmoid function to a layer
     * Compatible : Hidden, Output
     */
    for(int i = 0; i < l->size; i++) {
        l->neurons[i].activation = 1 / (1 + (exp(-l->neurons[i].activation)));
    }
}

void sigmoid_derivative_output(
        Layer *l,
        double (*cost)(double, double),
        const double *expectedOut
        )
{
    /*
     * Sigmoid derivative function for backpropagation
     * This function is made to work on the output layer only.
     */
    for(int i = 0; i < l->size; i++) {
        l->neurons[i].delta =
                cost(l->neurons[i].activation, expectedOut[i])
                * (l->neurons[i].activation) * (1 - l->neurons[i].activation);
    }
}

void sigmoid_derivative_hidden(Layer *l)
{
    /*
     * Sigmoid derivative function for backpropagation
     * This function is made to work on the hidden layers only.
     */
    for(int i = 0; i < l->size; i++)
        l->neurons[i].delta =
                (l->neurons[i].activation) * (1 - l->neurons[i].activation);
}

void relu(Layer *l)
{
    /*
     * Applies the relu function to a layer
     * Compatible : Hidden
     */
    for(int i = 0; i < l->size; i++)
        l->neurons[i].activation =
                l->neurons[i].activation < 0 ? 0 : l->neurons[i].activation;
}

void relu_derivative(Layer *l)
{
    /*
     * Relu derivative function for backpropagation
     * This function is made to work on the hidden layers only.
     */
    for(int i = 0; i < l->size; i++){
        l->neurons[i].delta = (l->neurons[i].activation>=0)? 1 : 0;
    }
}

void softmax(Layer *l)
{
    /*
     * Applies the softmax function ta a layer
     * Compatible : Output
     */

    double max = l->neurons[0].activation;
    for (int i = 1; i < l->size; i++) {
        if (l->neurons[i].activation > max) {
            max = l->neurons[i].activation;
        }
    }

    double sum = 0;
    for (int i = 0; i < l->size; i++) {
        // Subtract max for numerical stability
        l->neurons[i].activation = exp(l->neurons[i].activation - max);
        sum += l->neurons[i].activation;
    }

    for (int i = 0; i < l->size; i++) {
        l->neurons[i].activation /= sum;
        l->neurons[i].delta = 1;
    }
}

void cross_entropy(
        Layer *l,
        double(*cost)(double, double) __attribute__((unused)),
        const double* expectedOut
        )
{
    double error = 0.0;
    for (int i = 0; i < l->size; i++)
    {
        l->neurons[i].delta = l->neurons[i].activation - expectedOut[i];
        error += l->neurons[i].delta;
    }
    printf("error = %f\n", error*error);
}

/*
 * Error
 */
double Layer_getErrorTotal(const Layer* l)
{
    double total = 0;
    for (int i = 0; i < l->size; i++) {
        double e = l->neurons[i].error;
        total += e*e;
    }
    return (total / l->size);
}

/*
 * Initialisation functions of the neural network
 */

Neuron initNeuron(int nextLayerSize)
{
    /*
     * Initialise a new neuron
     */
    Neuron newNeuron =
            {
            0,
            (double*)malloc(nextLayerSize * sizeof(double)),
            0,
            0,
            0,
            (double*) calloc(nextLayerSize, sizeof(double)),
            0
            };
    return newNeuron;
}

Layer initLayer(int layerSize, int nextLayerSize)
{
    /*
     * Initialise a new layer
     */
    if(layerSize == 0)
        errx(EXIT_FAILURE, "initLayer : Cannot initialize a layer of size 0");

    Layer newLayer;
    newLayer.size = layerSize;
    newLayer.neurons = malloc(layerSize * sizeof(Neuron));

    // We do not want to initialise the neuron if there is no layer after
    for (int neuron = 0; neuron < layerSize; neuron++) {
        newLayer.neurons[neuron] = initNeuron(nextLayerSize);
    }

    return newLayer;
}

void initWeights(NeuralNetwork network)
{
    /*
     * Initialise random weights in the network
     * for every layer except the output layer
     */

    if(network.layers == NULL)
        errx(EXIT_FAILURE, "initWeights : The neural network has no layers");

    for(int i = 0; i < network.layerCount-1; i++)
    {
        for(int j = 0; j < network.layers[i].size; j++)
        {
            for (int k = 0; k < network.layers[i+1].size; k++) {
                network.layers[i].neurons[j].weights[k] = normalRand()*0.1;
            }
        }
    }
}

void initBiases(NeuralNetwork network)
{
    /*
     * Initialise random biases in the network
     * for every layer except the input layer
     */
    if(network.layers == NULL)
        errx(EXIT_FAILURE, "initBiases : The neural network has no layers");

    for(int i = 0; i < network.layerCount; i++)
    {
        for (int j = 0; j < network.layers[i].size; j++) {
            network.layers[i].neurons[j].bias = randD();
        }
    }
}

NeuralNetwork initNetwork(
        int layerCount,
        int* layerSizes,
        char* hidden_activation,
        char* output_activation
        )
{
    /*
     * Create and return a new neural network,
     * Arguments:
     * layerCount is the number of layer in the neural network
     * layerSize is a list of the size of each layer
     *      ex : {2,4,4,1}
     * nb:
     * layerSize must have the same as layerCount
     *
     * The hidden_activation can be
     *  - sigmoid
     *  - relu
     *
     * The output_activation can be
     *  - sigmoid
     *  - softmax
     */
    if(layerCount < 2)
        errx(EXIT_FAILURE,
             "initNetwork : The neural network must have more than 2 layers");

    NeuralNetwork newNetwork;
    newNetwork.layerCount = layerCount;
    newNetwork.layers = malloc(layerCount * sizeof(Layer));

    for(int layer = 0; layer < layerCount; layer++)
    {
        if(layerSizes[layer] < 1)
            errx(EXIT_FAILURE,
                 "initNetwork : One of the give layerSize is incorrect");
        newNetwork.layers[layer] = initLayer(
                layerSizes[layer],
                layer < layerCount-1 ? layerSizes[layer + 1] : 0
        );
    }

    initWeights(newNetwork);
    //initBiases(newNetwork);

    if(strcmp(hidden_activation, "relu") == 0)
    {
        newNetwork.hidden_activation = relu;
        newNetwork.hidden_activation_derivative = relu_derivative;
    }

    else if(strcmp(hidden_activation, "sigmoid") == 0)
    {
        newNetwork.hidden_activation = sigmoid;
        newNetwork.hidden_activation_derivative = sigmoid_derivative_hidden;
    }
    else
        errx(EXIT_FAILURE, "initNetwork : wrong hidden activation");

    if(strcmp(output_activation, "sigmoid") == 0)
    {
        newNetwork.output_activation = sigmoid;
        newNetwork.output_activation_derivative = sigmoid_derivative_output;
    }

    else if(strcmp(output_activation, "softmax") == 0)
    {
        newNetwork.output_activation = softmax;
        newNetwork.output_activation_derivative = cross_entropy;
    }
    else
        errx(EXIT_FAILURE, "initNetwork : wrong output activation");

    newNetwork.cost = MSE;
    return newNetwork;
}

void freeNetwork(NeuralNetwork net)
{
    /*
     * Free all the data allocated to the network
     */

    for(int layer = 0; layer < net.layerCount; layer++)
    {
        Layer l = net.layers[layer];
        for (int neuron = 0; neuron < l.size; neuron++) {
            free(l.neurons[neuron].weights);
            free(l.neurons[neuron].wdelta);
        }
        free(l.neurons);
    }
    free(net.layers);
}

/*
 * Network execution
 */

void feedInput(NeuralNetwork net, const double* input)
{
    /*
     * Set the input layer to the correct input
     * the input must have the same size as the
     * input layer.
     */
    if(net.layers == NULL)
        errx(EXIT_FAILURE,
             "feedInput : The given neural network does not have any layers");


    Layer inputLayer = net.layers[0];
    for(int i = 0; i < inputLayer.size; i++)
    {
        inputLayer.neurons[i].activation = input[i];
    }
}

void feedForward(NeuralNetwork net)
{
    /*
     * Feed the input layer through the neural network
     * up to the Output layer.
     */
    // Hidden layers
    for (int i = 1; i < net.layerCount - 1; i++)
        for (int j = 0; j < net.layers[i].size; j++)
        {
            net.layers[i].neurons[j].activation =
                    net.layers[i].neurons[j].bias;
            for (int k = 0; k < net.layers[i - 1].size; k++)
                net.layers[i].neurons[j].activation +=
                        (net.layers[i-1].neurons[k].weights[j]
                        * net.layers[i-1].neurons[k].activation);

            net.layers[i].neurons[j].activation =
                    sSigmoid(net.layers[i].neurons[j].activation);
            net.layers[i].neurons[j].delta =
                    (net.layers[i].neurons[j].activation)
                    *(1.0-net.layers[i].neurons[j].activation);
        }
    int last = net.layerCount - 1;
    for (int j = 0; j < net.layers[last].size; j++)
    {
        net.layers[last].neurons[j].activation =
                net.layers[last].neurons[j].bias;
        for (int k = 0; k < net.layers[last - 1].size; k++)
            net.layers[last].neurons[j].activation +=
                    net.layers[last - 1].neurons[k].weights[j]
                    * net.layers[last - 1].neurons[k].activation;
    }
    softmax(&net.layers[last]);
}

double* retrieveOutput(NeuralNetwork net)
{
    /*
     * Return the current Output of the neural network
     * Output is malloced
     */
    Layer OutputLayer = net.layers[net.layerCount-1];
    double* result = malloc(OutputLayer.size * sizeof(double));
    for (int neuron = 0; neuron < OutputLayer.size; neuron++) {
        result[neuron] = OutputLayer.neurons[neuron].activation;
    }
    return result;
}

double* execute(NeuralNetwork net, double* input)
{
    /*
     * Run the neural network
     * The input size must cover the whole input layer
     */

    // Set the input layer values
    feedInput(net, input);
    // Run the values through the network
    feedForward(net);
    // Retrieve the Output
    return retrieveOutput(net);
}
/*
 * Network training
 */
void update_weights(NeuralNetwork net, double learning_rate)
{
    /*
     * Update every weight of the neural network with
     * the stored delta which should be determined
     * by the backpropagation function.
     */
    for (int layer = net.layerCount - 1; layer > 0; layer--)
    {
        Layer *current = &net.layers[layer];
        Layer *prev = &net.layers[layer-1];
        for(int i = 0; i < current->size; i++) {
            current->neurons[i].bias -=
                    current->neurons[i].bdelta * learning_rate;
            current->neurons[i].bdelta = 0;
            for (int j = 0; j < net.layers[layer - 1].size; j++) {
                prev->neurons[j].weights[i] -=
                        prev->neurons[j].wdelta[i] * learning_rate;
                prev->neurons[j].wdelta[i] = 0;
            }
        }
    }
}

void backPropagation(NeuralNetwork net, const double* expectedOut)
{
    /*
     * Back-propagate and updates the biases and weights
     * into the give neural network
     */

    int currentIndex = net.layerCount - 1;
    Layer *current = &net.layers[currentIndex];
    Layer *prev = &net.layers[currentIndex-1];
    // Run error
    for (int i = 0; i < current->size; i++)
    {
        current->neurons[i].error =
                (current->neurons[i].activation - expectedOut[i]);
    }

    while(currentIndex > 0) {
        current = &net.layers[currentIndex];
        prev = &net.layers[currentIndex-1];
        // Clear the previous errors;
        for (int j = 0; j < prev->size; j++)
            prev->neurons[j].error = 0;
        for (int i = 0; i < current->size; i++) {
            // Compute the weight / bias updates
            double delta =
                    current->neurons[i].error * current->neurons[i].delta;
            for (int j = 0; j < prev->size; j++) {
                prev->neurons[j].error +=
                        prev->neurons[j].weights[i] * delta;
                prev->neurons[j].wdelta[i] +=
                        prev->neurons[j].activation * delta;
            }
            current->neurons[i].bdelta += delta;
        }
        currentIndex--;
    }
}

void training(NeuralNetwork net,
              double*** dataSet,
              int dataSize,
              double learningRate,
              int epoch,
              int batchsize
)
{
    /*
     * Train the neural network on the give dataSet
     *
     * the dataSet must follow the following form
     * {
     *  { InputData, OutputData},
     *  { InputData, OutputData},
     *  ...
     *  }
     *
     *  and dataSize is the number of different set inside the dataSet
     */
    double eTotal = 0;
    printf("TRAINING ! ITERATIONS = %i\n", epoch);
    for(int iterations = 0; iterations <= dataSize*epoch; iterations++)
    {
        int randIN = rand()%dataSize;
        double* input = dataSet[randIN][0];
        double* expectedOutput = dataSet[randIN][1];
        //printDATA((const double**) dataSet[randIN]);
        feedInput(net, input);
        feedForward(net);
        backPropagation(net, expectedOutput);
        eTotal += Layer_getErrorTotal(&net.layers[net.layerCount-1]);;
        if((iterations%batchsize) == 0)
            update_weights(net, learningRate);
        if(iterations % 1000 == 0)
        {
            printf("%i; %.6f\n", iterations, eTotal/1000);
            eTotal=0;
        }
    }
}

void saveNeuralNetwork(NeuralNetwork net, const char *filePath) {
    /*
     * Dump the given neural network into the given file
     * Format :
     *
     * {LayerCount}{Layer0.size,Layer1.size,...}
     * {Neurons data}
     *
     * The following properties of the neurons are not being saved
     * and considered non-essential once the training is done
     * they should be set to 0 when loading the network :
     *  - activation
     *  - *_delta
     */

    FILE *file = fopen(filePath, "wb");
    if (file == NULL)
        err(EXIT_FAILURE, "saveNeuralNetwork : Error while opening file");

    // Save the number of layers
    fwrite(&net.layerCount, sizeof(int), 1, file);
    for (int layer = 0; layer < net.layerCount; layer++) {
        // Save the size of the layer
        fwrite(&net.layers[layer].size, sizeof(int), 1, file);
    }

    for (int layer = 0; layer < net.layerCount; layer++)
    {
        for (int neuron = 0; neuron < net.layers[layer].size; ++neuron) {
            Neuron *current = &net.layers[layer].neurons[neuron];
            // Save the current base properties
            fwrite(&current->bias, sizeof(double), 1, file);
            if(layer < net.layerCount - 1) // Final layer does not have biases
                fwrite(
                        current->weights,
                        sizeof(double),
                        net.layers[layer + 1].size,
                        file
                );
        }
    }

    fclose(file);
}

NeuralNetwork loadNeuralNetwork(const char *filePath) {
    /*
     * Load the previously saved neural network from the given file
     */
    FILE *file = fopen(filePath, "rb");
    if (file == NULL)
        err(EXIT_FAILURE, "loadNeuralNetwork : Error while reading file");

    NeuralNetwork net;

    // Load the number of layers
    fread(&net.layerCount, sizeof(int), 1, file);
    net.layers = malloc(net.layerCount * sizeof(Layer));

    // Load layer sizes and initiate layers
    for (int layer = 0; layer < net.layerCount; ++layer) {
        fread(&net.layers[layer].size, sizeof(int), 1, file);
        net.layers[layer].neurons =
                malloc(net.layers[layer].size * sizeof(Neuron));
    }

    // Load neurons data
    for (int layer = 0; layer < net.layerCount; layer++) {
        // Iterate over neurons
        for (int neuron = 0; neuron < net.layers[layer].size; neuron++) {
            Neuron *current = &net.layers[layer].neurons[neuron];

            // Load the current base properties
            fread(&current->bias, sizeof(double), 1, file);
            if(layer < net.layerCount-1) {
                current->weights =
                        malloc(net.layers[layer + 1].size * sizeof(double));
                fread(
                        current->weights,
                        sizeof(double),
                        net.layers[layer + 1].size,
                        file
                );
                current->wdelta =
                        calloc(net.layers[layer+1].size, sizeof(double));
            }
            else {
                current->wdelta = NULL;
                current->weights = NULL;
            }

            // Set the deltas and activation to 0
            current->delta = 0;
            current->activation = 0;
            current->bdelta = 0;
            current->error = 0;
        }
    }

    fclose(file);
    return net;
}
