//
// Created by martin on 07/11/23.
//
#include "neuralnetwork.h"
#include <string.h>
#include "err.h"
#include <stdio.h>
#include <stdlib.h>
#include "idx.h"
#include "convert.h"
#include "dirent.h"


#define SIZES {784, 80, 10}
#define LEARNINGRATE 0.01
#define ITERATIONS 10
#define OUTSIZE 10
#define BATCHSIZE 32

#define train_images "train-images-idx3-ubyte"
#define train_labels "train-labels-idx1-ubyte"

#define test_images "test-images-idx3-ubyte"
#define test_labels "test-labels-idx1-ubyte"

#define mapping "mapping.txt"



void printDATAX(const double* input)
{
    /*
     * This function is meant to print the images in the MNIST dataSet
     */
    printf("IMG>\n");
    for(int i = 0; i < 28; i++)
    {
        for(int j = 0; j < 28; j++)
        {
            char val =' ';
            if (input[j * 28 + i] > 0.3)
                val = '?';
            if (input[j * 28 + i] > 0.5)
                val = '@';
            if (input[j * 28 + i] > 0.8)
                val = '#';
            printf("%c ",val);
        }
        printf("\n");
    }
}

char* strccat(const char* s1, const char* s2)
{
    /*
     * Return new string without modifying s1 and s2;
     */
    char* x = malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(x, s1);
    strcat(x, s2);
    return x;
}

double*** genDataSet(double** Images, double** Labels, int *size)
{
    int i = 0;
    int deltaSize = 0;
    double ***data = malloc(*size * sizeof(double**));
    while(i < *size)
    {
        if(Labels[i])
        {
            data[i-deltaSize] = malloc(2*sizeof(double*));
            data[i-deltaSize][0] = Images[i];
            data[i-deltaSize][1] = Labels[i];
        }
        else // A value we are not interested to train on
        {
            free(Images[i]);
            deltaSize++;
        }
        i++;
    }
    *size -= deltaSize;
    // Reduce the size to the strict necessary
    return realloc(data, *size * sizeof(double**));
}

double*** genOcrDataSet(char* dataFolder, int outLayerSize, int *dataSize)
{
    char* Fmap = strccat(dataFolder, mapping);
    char* map = parseMapping(Fmap);
    free(Fmap);

    // Extract images
    int tImagesSize;
    char* FEIm = strccat(dataFolder, train_images);
    double **tImages = extractImages(FEIm, &tImagesSize);
    free(FEIm);

    // Extract labels
    int tLabelsSize;
    char* FELb = strccat(dataFolder, train_labels);
    double **tLabels = extractLabels(
            FELb,
            &tLabelsSize,
            outLayerSize,
            map
    );
    free(FELb);
    free(map);

    if (tImagesSize != tLabelsSize)
        errx(
                EXIT_FAILURE,
                "The number of labels(%i) and images(%i) is different",
                tLabelsSize,
                tImagesSize
                );

    double ***data = genDataSet(tImages, tLabels, &tImagesSize);
    *dataSize = tImagesSize;

    free(tImages);
    free(tLabels);
    return data;
}

double*** genOcrTestSet(char* dataFolder, int outLayerSize, int *dataSize)
{
    char* Fmap = strccat(dataFolder, mapping);
    char* map = parseMapping(Fmap);
    free(Fmap);

    // Extract images
    int tImagesSize;
    char* FEIm = strccat(dataFolder, test_images);
    double **tImages = extractImages(FEIm, &tImagesSize);
    free(FEIm);

    // Extract labels
    int tLabelsSize;
    char* FELb = strccat(dataFolder, test_labels);
    double **tLabels = extractLabels(
            FELb,
            &tLabelsSize,
            outLayerSize,
            map
    );
    free(FELb);
    free(map);

    if (tImagesSize != tLabelsSize)
        errx(
                EXIT_FAILURE,
                "The number of labels(%i) and images(%i) is different",
                tLabelsSize,
                tImagesSize
                );

    double ***data = genDataSet(tImages, tLabels, &tImagesSize);
    *dataSize = tImagesSize;

    free(tImages);
    free(tLabels);
    return data;
}

char argMax(const double* vect)
{
    char max = 0;
    for(char i = 1; i < OUTSIZE; i++)
    {
        if (vect[(int)max] < vect[(int)i])
            max = i;
    }
    return max;
}

void testSet(NeuralNetwork net, double*** test_set, int dataSize)
{
    printf("\n----- TESTING -----\n");
    int success = 0;
    for(int i = 0; i < dataSize; i++)
    {
        double* res = execute(net, test_set[i][0]);
        char got = argMax(res);
        char expected = argMax(test_set[i][1]);
        free(res);
        printf("Obtained %0i\tExpected %0i\t", got, expected);
        if (got == expected) {
            success++;
            printf("SUCCESS\n");
        }
        else
            printf("FAIL\n");
    }
    printf(
            "Success Rate = %i/%i (%f)\n",
            success,
            dataSize,
            (double)success/(double)dataSize*100
            );

}

void freeDataSet(double*** dataSet, int size)
{
    for(int i = 0; i < size; i++)
    {
        free(dataSet[i][0]);
        free(dataSet[i][1]);
        free(dataSet[i]);
    }
    free(dataSet);
}

SDL_Surface* loadImage(const char* imagePath)
{
    SDL_Surface *img;

    img = IMG_Load(imagePath);
    if (!img)
        errx(3, "can't load %s: %s", imagePath, IMG_GetError());

    return img;
}

int main(int argv, char** argc)
{
    if (argv < 2)
        errx(EXIT_FAILURE, "Not enough arguments");

    else if (strcmp(argc[1],"-h") == 0) // HELP
    {
        printf("--- Help ---\n"
               "ocr [PARAMETERS] {IN_FILE} {OUT_FILE}\n"
               "-h : Display this informations\n"
               "-g {NEURAL_FILE} {DATA_FOLDER} to generate a new network\n"
               "-t {NEURAL_FILE} {DATA_FOLDER} to test a neural network\n"
               );
    }

    else if(strcmp(argc[1],"-g") == 0 && argv == 4) // Generate neural network
    {
        char* neuralOut = argc[2];
        char* dataFolder = argc[3];
        int layerSizes[] = SIZES;
        int iterations = ITERATIONS;
        double learningRate = LEARNINGRATE;
        NeuralNetwork network = initNetwork(
                sizeof(layerSizes) / sizeof (int),
                layerSizes,
                "sigmoid",
                "softmax"
                );

        int dataSize;
        double ***dataSet = genOcrDataSet(
                dataFolder,
                network.layers[network.layerCount - 1].size,
                &dataSize
                );

        printf("Size of dataset = %i\n", dataSize);
        training(
                network, dataSet, dataSize, learningRate, iterations, BATCHSIZE
                );
        freeDataSet(dataSet, dataSize);

        dataSet = genOcrTestSet(
                dataFolder,
                network.layers[network.layerCount - 1].size,
                &dataSize
                );
        testSet(network, dataSet, dataSize);
        freeDataSet(dataSet, dataSize);
        saveNeuralNetwork(network, neuralOut);
        freeNetwork(network);
    }

    else if(strcmp(argc[1], "-t") == 0 && argv == 4) // Test network
    {
        NeuralNetwork ocr = loadNeuralNetwork(argc[2]);
        char* dataFolder = argc[3];
        int dataSize;
        double ***dataSet = genOcrTestSet(
                dataFolder,
                ocr.layers[ocr.layerCount - 1].size,
                &dataSize
                );
        testSet(ocr,dataSet, dataSize);
        freeDataSet(dataSet,dataSize);
        freeNetwork(ocr);
    }

    else if(strcmp(argc[1], "-b") == 0 && argv == 3)
    {
        NeuralNetwork ocr = loadNeuralNetwork(argc[2]);
        double *blank = calloc(ocr.layers[0].size, sizeof(double));
        memset(blank, 1, ocr.layers[0].size);
        double *result = execute(ocr, blank);
        for(int i = 0; i < ocr.layers[ocr.layerCount-1].size; i++)
            printf("%f ", result[i]);

        printf("\n");
        free(blank);
        free(result);
        freeNetwork(ocr);
    }
    else if(strcmp(argc[1], "-n") == 0 && argv == 5)
    {

        NeuralNetwork ocr = loadNeuralNetwork(argc[2]);
        char* inFolder = argc[3];
        char* outFile = argc[4];
        DIR *inDir = opendir(inFolder);
        if(inDir == NULL)
            errx(EXIT_FAILURE, "Wrong input folder");
        FILE *f = fopen(outFile, "w");
        if(f == NULL)
            errx(EXIT_FAILURE, "Failed to open outFile");
        char* sudoku = malloc(81 * sizeof(char));
        if(sudoku == NULL)
            errx(EXIT_FAILURE, "Not enough memory");
        memset(sudoku, '.', 81);
        struct dirent *dire;
        while((dire = readdir(inDir)) != NULL) {
            int x, y;
            if(dire->d_type != 8) {
                continue;
            }
            if (sscanf(dire->d_name, "Case_(%d_%d).png", &x, &y) != 2) {
                errx(EXIT_FAILURE, "Wrong name format");
            }
            if(x > 9 || y > 9)
                errx(EXIT_FAILURE, "Wrong coordinates");

            char *absName = strccat(inFolder, dire->d_name);
            SDL_Surface *img = loadImage(absName);
            double *_res = convert_image(img);
            //printDATAX(_res);
            double *_ocrRes = execute(ocr, _res);
            sudoku[((y-1) * 9 + (x-1))] = (char)(argMax(_ocrRes) + '0');
            SDL_FreeSurface(img);
            free(absName);
            free(_res);
            free(_ocrRes);
        }
        closedir(inDir);
        for(int i = 0; i < 9; i++)
        {
            for(int j = 0; j < 9; j++)
            {
                fprintf(f, "%c",sudoku[i*9 + j]);
                if((j+1)%3 == 0)
                    fprintf(f, " ");
            }
            fprintf(f, "\n");
            if((i+1)%3 == 0)
                fprintf(f, "\n");
        }
        fclose(f);
        freeNetwork(ocr);
        free(sudoku);
    }

    else {
        errx(EXIT_FAILURE, "Wrong parameters!");
    }
}
