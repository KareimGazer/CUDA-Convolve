
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define MASK_WIDTH 17 // most conv kernels (arrays) will not exceed 10 in each dimension
#define HALF_SPAN MASK_WIDTH/2

#define WIDTH 768
#define HEIGHT WIDTH
#define PITCH WIDTH

#define TILE_SIZE 32
#define OUTPUT_TILE_WIDTH 32


__constant__ float dev_mask[MASK_WIDTH * MASK_WIDTH];

__global__ void conv_2D(float* input_arr, float* output_arr, float missing_value);
__global__ void conv_2D_cached(float* input_arr, float* output_arr, float missing_value);

__global__ void Id(float* input_arr, float* output_arr) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    output_arr[idx_y * WIDTH + idx_x] = idx_y * WIDTH + idx_x;
}

float mask_org[MASK_WIDTH] = { 0.00015,	0.00081, 0.00344, 0.01169, 0.03179,	0.06918, 0.12058, 0.16829, 0.18806, 0.16829, 0.12058, 0.06918, 0.03179, 0.01169, 0.00344, 0.00081, 0.00015 };
int main()
{
    // converting the kernel into 2D
    float mask_arr[MASK_WIDTH * MASK_WIDTH] = {0};
    for (int i = 0; i < MASK_WIDTH; i++) {
        for (int j = 0; j < MASK_WIDTH; j++) {
            mask_arr[i * MASK_WIDTH + j] = mask_org[i] * mask_org[j];
        }
    }
    FILE* input_file_ptr;
    input_file_ptr = fopen("ind9.bin", "rb");
    float* input = (float*)calloc(HEIGHT * WIDTH, sizeof(float));
    fread(input, sizeof(float), WIDTH * WIDTH, input_file_ptr);

    float* output = (float*)calloc(HEIGHT * WIDTH, sizeof(float));

    float* dev_input = 0; float* dev_output = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpyToSymbol(dev_mask, mask_arr, sizeof(float) * MASK_WIDTH * MASK_WIDTH); // copy to constant memory

    cudaMalloc((void**)&dev_input, HEIGHT * WIDTH * sizeof(float));
    cudaMalloc((void**)&dev_output, HEIGHT * WIDTH * sizeof(float));

    cudaMemcpy(dev_input, input, HEIGHT * WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(WIDTH / 32.0), ceil(HEIGHT / 32.0), 1);
    dim3 dimBlock(32, 32, 1);

    cudaEventRecord(start);
    conv_2D<<<dimGrid, dimBlock>>>(dev_input, dev_output, 0);
    cudaEventRecord(stop);

    cudaMemcpy(output, dev_output, HEIGHT * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    FILE* output_file_ptr;
    output_file_ptr = fopen("output.txt", "w");
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            // if(output[i * WIDTH + j] != 0)
                fprintf(output_file_ptr, "%.5f \t", output[i * WIDTH + j]);
        }
        fprintf(output_file_ptr, "\n");
    }
    printf("Time spent: %f ms\n", milliseconds);

    cudaFree(dev_input);
    cudaFree(dev_output);

    free(input);
    free(output);

    fclose(input_file_ptr);
    fclose(output_file_ptr);
    return 0;
}

__global__ void conv_2D(float* input_arr, float* output_arr, float missing_value) {
    int row_index = blockIdx.y * blockDim.y + threadIdx.y;
    int col_index = blockIdx.x * blockDim.x + threadIdx.x;

    int window_start_x = col_index - HALF_SPAN;
    int window_start_y = row_index - HALF_SPAN;
    // to handel boundries
    float accumulator = 0;
    int current_row_index;
    int current_col_index;
    for (int i = 0; i < MASK_WIDTH; i++) {
        for (int j = 0; j < MASK_WIDTH; j++) {
            current_row_index = window_start_y + i;
            current_col_index = window_start_x + j;
            if (window_start_y >= 0 && window_start_x >= 0 && window_start_y < HEIGHT && window_start_x < WIDTH) {
                accumulator += input_arr[current_row_index * WIDTH + current_col_index] * dev_mask[i * MASK_WIDTH + j];
            }
            else {
                accumulator += missing_value * dev_mask[i * MASK_WIDTH + j];
            }
        }
    }
    output_arr[row_index * WIDTH + col_index] = accumulator;
}

