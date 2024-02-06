//
// Created by martin on 12/7/23.
//
#include "convert.h"
#include <stdlib.h>
#include <err.h>

Uint8 *pixel_ref(SDL_Surface *surf, unsigned x, unsigned y)
{
    int bpp = surf->format->BytesPerPixel;
    return (Uint8 *)surf->pixels + y * surf->pitch + x * bpp;
}

Uint32 get_pixel(SDL_Surface *surface, unsigned x, unsigned y)
{
    Uint8 *p = pixel_ref(surface, x, y);

    switch (surface->format->BytesPerPixel)
    {
        case 1:
            return *p;

        case 2:
            return *(Uint16 *)p;

        case 3:
            if (SDL_BYTEORDER == SDL_BIG_ENDIAN)
                return p[0] << 16 | p[1] << 8 | p[2];
            else
                return p[0] | p[1] << 8 | p[2] << 16;

        case 4:
            return *(Uint32 *)p;
    }

    return 0;
}

//preprocess image
double* convert_image(SDL_Surface* image) {
    // Assuming the image is grayscale (1 channel) and 28x28 pixels
    if (image->w != 28 || image->h != 28) {
        // Handle incorrect image dimensions or format
        errx(EXIT_FAILURE, "Invalid image dimensions or format");
    }

    double* result = malloc(784 * sizeof(double));
    if (result == NULL)
        errx(EXIT_FAILURE, "Not enough memory");

    // Iterate through image pixels and normalize values to the range [0, 1]
    for (int x = 0; x < 28; x++) {
        for (int y = 0; y < 28; y++) {
            Uint8 r, g, b;
            Uint32 pixel_value = get_pixel(image, x, y);
            SDL_GetRGB(pixel_value, image->format, &r, &g, &b);
            double grayscale = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            result[x * 28 + y] = grayscale / 255.0;
        }
    }

    return result;
}