//
// Created by martin on 09/11/23.
//
#pragma once

char* parseMapping(char* filename);

void freeMat(double** ptr, int size);

/*
 * Extract images from the dataset
 */
double** extractImages(char* filename, int *size);

/*
 * Extract labels from the dataset
 */
double** extractLabels(char* filename, int *size, int outSize, char* map);
