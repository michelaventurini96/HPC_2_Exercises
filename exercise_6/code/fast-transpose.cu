/*Ex. 5
1. implement matrix transpose in cuda with shared memory
  (use block algorithm/FAST TRANSPOSE).
2. MATRIX SIZE = 8192X8192 and register the time of solution compared with
  naive one for 64,512, 1024 threads/block.
3. Try to reach mem band = 100 Gb/s.
*/
#include <stdio.h>
// kernels transpose a tile of TILE_DIM x TILE_DIM elements
// using a TILE_DIM x BLOCK_ROWS thread block, so that each thread
// transposes TILE_DIM/BLOCK_ROWS elements. TILE_DIM is an
// integral multiple of BLOCK_ROWS
#define TILE_DIM 32
//#define BLOCK_ROWS 2 //64 threads/block
//#define BLOCK_ROWS 16 //512 threads/block
#define BLOCK_ROWS 32 //1024 threads/block
// Number of repetitions used for timing.
#define NUM_REPS 100

__host__ void printMatrix(const float* const data, const size_t size_x, const size_t size_y, const size_t a, const size_t b){
  for(size_t i=0; i<a; i++){
  	printf("%5.2f ", data[i]);
  }
  printf("\n");
}

__host__ int compareRes(const float* const odata, const float* const gold, const size_t msize){
  int res = 0;
  for(size_t i = 0; i<msize; i++){
    if (odata[i] != gold[i]) ++res;
  }
  return res;
}

__global__ void transposeNaive(float *odata, const float* const idata,
    const int width, const int height) {
  int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;
  int index_in = xIndex + width * yIndex;
  int index_out = yIndex + height * xIndex;

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
        odata[index_out+i] = idata[index_in+i*width];
    }
}

__host__ void computeTransposeHost(float* odata, const float* const h_idata,
    const int size_x, const int size_y){
  for (size_t i = 0; i < size_x; i++) {
    for (size_t j = 0; j < size_y; j++) {
      odata[j+i*size_x] = h_idata[i + j*size_y];
    }
  }
}

__global__ void transposeCoalesced(float *odata, const float *const idata, const int width,
    const int height){
  __shared__ float tile[TILE_DIM][TILE_DIM+1]; //padding so different threads access different shared memory banks and no bottleneck

  int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) { // use a tile like a buffer to transpose blocks
      tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
  }

  __syncthreads();
  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
      odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
  }
}

int main( int argc, char** argv) {
  
// set matrix size
  const int size_x = 8192;
  const int size_y = 8192;

// execution configuration parameters
  dim3 grid(size_x/TILE_DIM, size_y/TILE_DIM), threads(TILE_DIM,BLOCK_ROWS);
  
// CUDA events
  cudaEvent_t start, stop;
  
// size of memory required to store the matrix
  const int mem_size = sizeof(float) * size_x*size_y;
  
// allocate host memory
  float *h_idata = (float*) malloc(mem_size);
  float *h_odata = (float*) malloc(mem_size);
  float *transposeHost = (float *) malloc(mem_size);
  
// allocate device memory
  float *d_idata, *d_odata;
  cudaMalloc( (void**) &d_idata, mem_size);
  cudaMalloc( (void**) &d_odata, mem_size);
  
// initalize host data
  for(int i = 0; i < (size_x*size_y); ++i) 
	h_idata[i] = (float) i;
  printf("Initial matrix:\n");
  printMatrix(h_idata, size_x, size_y, 10, 10);
 
// copy host data to device
  cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice );

// Compute reference transpose solution
  computeTransposeHost(transposeHost, h_idata, size_x, size_y);
  printf("Reference solution:\n");
  printMatrix(transposeHost, size_x, size_y, 10, 10);


  printf("\nMatrix size: %dx%d, tile: %dx%d, block: %dx%d\n\n",
  	size_x, size_y, TILE_DIM, TILE_DIM, TILE_DIM, BLOCK_ROWS);
  printf("Kernel\t\t\tLoop over kernel\n");
  printf("------\t\t\t----------------\n");

  
// initialize events, EC parameters
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

//*********************************time transposeNaive*************************************
  cudaEventRecord(start, 0);
  for (int i=0; i < NUM_REPS; i++) {
    transposeNaive<<<grid, threads>>>(d_odata, d_idata,size_x,size_y);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float outerTime;
  cudaEventElapsedTime(&outerTime, start, stop);

  cudaMemcpy(h_odata,d_odata, mem_size, cudaMemcpyDeviceToHost);
  
//check correctness of solution
  int res = compareRes(transposeHost, h_odata, size_x*size_y);
  if (res != 0) printf("*** transposeNaive kernel FAILED ***\n");
  else printf("***transposeNaive kernel OK***\n");
// report effective bandwidths
  float outerBandwidth = 2.*1000*mem_size/(1024*1024*1024)/(outerTime/NUM_REPS);
  
  printf("transposeNaive\t\t%5.2f GB/s\n", outerBandwidth);
  //printf("transposeNaive:\n");
  //printMatrix(h_odata, size_x, size_y, 10, 10);

//*********************************time transposeCoalesced************************************
  cudaEventRecord(start, 0);
  for (int i=0; i < NUM_REPS; i++) {
      transposeCoalesced<<<grid, threads>>>(d_odata, d_idata,size_x,size_y);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  //float outerTime1;
  cudaEventElapsedTime(&outerTime, start, stop);
  cudaMemcpy(h_odata,d_odata, mem_size, cudaMemcpyDeviceToHost);
  
//check corrctness of result
  res = compareRes(transposeHost, h_odata, size_x*size_y);
  if (res != 0) printf("*** transposeCoalasced kernel FAILED ***\n");
  else printf("***transposeCoalasced kernel OK***\n");
// report effective bandwidths
  outerBandwidth = 2.*1000*mem_size/(1024*1024*1024)/(outerTime/NUM_REPS);
    
  printf("transposeCoalesced\t%5.2f GB/s\n", outerBandwidth);
  //printf("transposeCoalesced\n");
  //printMatrix(h_odata, size_x, size_y, 10, 10);
  
// cleanup memory
  free(h_idata); free(h_odata); free(transposeHost);
  cudaFree(d_idata); cudaFree(d_odata);
  cudaEventDestroy(start); cudaEventDestroy(stop);

  return 0;
}
