// Extract raw data from the dataset using idx
// Created by martin on 12/6/23.
//

#include "idx.h"
#include <stdlib.h>
#include <stdio.h>
#include <err.h>

#define OUTPUT "out.txt"

int main(int argc, char **argv)
{
    if (argc < 3)
        errx(EXIT_FAILURE, "Argument unspecified");

    char* file = argv[1];
    int size = atoi(argv[2]);
    FILE *fd = fopen(OUTPUT, "w");
    int im_size;
    double **images = extractImages(file, &im_size);
    for(int i = 0; i < size; i++)
    {
        fprintf(fd, "{");
        for(int j = 0; j < 783; j++)
            fprintf(fd,"%f,", images[i][j]);
        fprintf(fd, "%f}\n", images[i][783]);
    }

    fclose(fd);
    for(int i = 0; i < im_size; i++)
        free(images[i]);
    free(images);
}