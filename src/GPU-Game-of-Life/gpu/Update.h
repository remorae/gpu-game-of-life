#ifndef UPDATE_H
#define UPDATE_H

void updateGridOnGPU(unsigned char* grid,
                     size_t gridWidth, size_t gridHeight,
                     size_t blockWidth, size_t blockHeight);

#endif