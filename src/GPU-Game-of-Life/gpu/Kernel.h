#ifndef KERNEL_H
#define KERNEL_H

__global__ void updateGrid(unsigned char* destGrid, unsigned char* srcGrid, int width, int height, int offsetX, int offsetY);

#endif