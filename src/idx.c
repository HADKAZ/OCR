//
// Created by martin on 09/11/23.
//
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "idx.h"

#define DEBUG 0

int isLittleEndian() {
    /*
     * Check if the system interprets numbers as big endian
     * or little endian for data extraction
     */

    // Use a union to interpret the same memory location as different types
    union {
        int i;
        char c[sizeof(int)];
    } test;

    // Set the integer value to 1
    test.i = 1;

    // If the first byte of the integer is 1, the system is little-endian
    return (test.c[0] == 1);
}

int reverseEndian(int num) {
    // Use bitwise operations to reverse the endianness
    return ((num >> 24) & 0xFF) |
           ((num >> 8) & 0xFF00) |
           ((num << 8) & 0xFF0000) |
           ((num << 24) & 0xFF000000);
}

void freeMat(double** ptr, int size)
{
    for(int i = 0; i < size; i++)
        free(ptr[i]);
    free(ptr);
}

double** extractImages(char* filename, int *size)
{
    /*
     * This will extract the images of the dataset
     */
    FILE* file = fopen(filename, "rb");
    int magicNumber;
    fread(&magicNumber, 4, 1, file);

    fread(size, 4, 1, file);

    int rows;
    fread(&rows, 4, 1, file);

    int cols;
    fread(&cols, 4, 1, file);

    if(isLittleEndian()) // Fix for intel processor LittleEndian
    {
        magicNumber = reverseEndian(magicNumber);
        *size = reverseEndian(*size);
        rows = reverseEndian(rows);
        cols = reverseEndian(cols);
    }

    if(DEBUG)
        printf(
                "[%s] Debug information\n- Magic number = %0i\n- size = %i\n- rows = %i\n- cols = %i\n",
                filename, magicNumber, *size, rows, cols);
    double **images = malloc(*size*sizeof(double*));
    unsigned char* image = malloc(rows*cols);
    for(int i = 0; i < *size; i++)
    {
        fread(image, 1, rows*cols, file);
        images[i] = malloc(rows*cols*sizeof(double));
        for(int j = 0; j < rows*cols; j++)
        {
            images[i][j] = image[j] / 255.0;
        }
    }

    free(image);
    return images;
}

char toHex(unsigned char input)
{
    if (input > '0' && input <= '9')
        return input - '0';

    if (input >= 'a' && input <= 'f')
        return input + 10 - 'a';

    if (input >= 'A' && input <= 'F')
        return input + 10 - 'A';

    return -1;
}

double* toInput(unsigned char letter, int outSize, char *map)
{
    char conv = toHex(map[letter]);
    if (conv != -1) {
        double *vect = calloc(outSize, sizeof(double));
        vect[(int)conv] = 1.0;
        return vect;
    }
    return NULL;
}

double** extractLabels(char* filename, int *size, int outSize, char *map)
{
    /*
     * This will extract the labels of the dataset
     */

    FILE* file = fopen(filename, "rb");
    if(!file)
        errx(EXIT_FAILURE, "extractLabels : file not found (%s)",filename);
    int magicNumber;
    fread(&magicNumber, 4, 1, file);
    fread(size, 4, 1, file);

    if(isLittleEndian()) // Fix for intel processor LittleEndian
    {
        magicNumber = reverseEndian(magicNumber);
        *size = reverseEndian(*size);
    }

    if(DEBUG)
        printf(
                "[%s] Debug information\n- Magic number = %0i\n- size = %i\n",
                filename, magicNumber, *size);

    double **labels = malloc(*size*sizeof(double*));
    for(int i = 0; i < *size; i++)
    {
        unsigned char label;
        fread(&label, 1, 1, file);
        labels[i] = toInput(label, outSize, map);
    }

    return labels;
}

char* parseMapping(char* filename) {
    FILE* file = fopen(filename, "r");

    if (!file)
        errx(EXIT_FAILURE, "parseMapping : error while opening file");

    char line = 0;
    char* map = NULL;
    int index, value;
    while (fscanf(file, "%i %i", &index, &value) == 2) {
        if(line != index)
            errx(EXIT_FAILURE, "parseMapping : invalid first value");

        line++;
        map = realloc(map, line);
        map[line-1] = (char)value;
    }

    fclose(file);

    return map;
}