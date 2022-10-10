#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <vector> // Для векторов
#include <algorithm> // Для generate

#include <cuda_runtime.h> 

// Для работы curand
#include <curand.h>
#include <curand_kernel.h>

using std::vector;
using std::generate;
using std::cout;
using std::cerr;
using std::endl;


// Конструкция проверок на ошибки
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
    }
}

// Генерация случайных чисел с библиотекой curand
__device__ float generate(curandState* globalState){
    int ind = threadIdx.x;
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState;
    return RANDOM;
}

// Для работы с curand необходимо выполнить setup ядра
__global__ void setup_kernel ( curandState * state, unsigned long seed ){
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
}

__global__ void randomN(int N, int *y, curandState* globalState) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
     while (tid < N)
     {
        int number = generate(globalState) * 1000000;
        if (N < 10)
            printf(" %i,", number);
        // Атомарная операция добавления в массив, позволяет обеспечить безопасность данных 
        // посредством блока адреса, куда записывает данные
        atomicAdd(&(y[tid]), number);
        tid += blockDim.x * gridDim.x;
    }
}

// C++11 way генерация случайных чисел с нормальным распределением
void testCpu(const int N, int *y) {
    long low_dist  = 0;
    long high_dist = 1000000;
    std::srand( ( unsigned int )std::time( nullptr ) );
    for( int i = 0; i < N; ++i ){
      y[i] = low_dist + std::rand() % ( high_dist - low_dist );
      if (N < 10)
        std::cout << " " <<low_dist + std::rand() % ( high_dist - low_dist ) << ",";
    }
    cout << endl;
}


int main() {

  using clock = std::chrono::system_clock;
  using sec = std::chrono::duration<double, std::milli>;

  constexpr int N = 2000000000;

  int *y, *d_y;
  y = (int*)malloc(N*sizeof(int));

  // Количество тредов на блок
  int NUM_THREADS = 1024;

  // Рассчет минимально необходимого количества блоков.
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
  //int NUM_BLOCKS = 2;

  CHECK_CUDA_ERROR(cudaMalloc(&d_y, N * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_y, y, N * sizeof(int), cudaMemcpyHostToDevice));

  curandState* devStates;
  cudaMalloc (&devStates, N * sizeof(curandState));
  // Генерируем сид
  std::srand( ( unsigned int )std::time( nullptr ) );
  int seed = rand();

  setup_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(devStates,seed);

  // Создаем эвенты, которые рассчитают время выполнения на GPU.
  cudaEvent_t start,stop;
  float gpuTime = 0.0f;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  CHECK_CUDA_ERROR(cudaEventRecord(start,0));

  cout << endl << "CUDA random numbers" << endl;
  // Запускаем в работу GPU асинхронно
  randomN<<<NUM_BLOCKS, NUM_THREADS>>>(N, d_y, devStates);

  CHECK_CUDA_ERROR(cudaEventRecord(stop,0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpuTime, start, stop));
  CHECK_CUDA_ERROR(cudaMemcpy(y, d_y, N*sizeof(int), cudaMemcpyDeviceToHost));

  // Уничтожаем созданные эвенты
  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));

  CHECK_LAST_CUDA_ERROR();
  // ----------------------------

  // Вывод массива GPU
  cout << "Last 10 numbers from GPU array: ";
  for (int i = N-1; i >= N-10; i--)
    cout << y[i] << ",";
  cout << endl;

  // Тестируем время выполнения на CPU
  cout << endl << "CPU random numbers" << endl;
  const auto before = clock::now();

  testCpu(N,y);

  const sec duration = clock::now() - before;

  // Вывод массива CPU
  cout << "Last 10 numbers from CPU array: ";
  for (int i = N-1; i >= N-10; i--)
    cout << y[i] << ",";
  cout << endl;

  // Вывод времени

  cout << "Time elapsed on GPU: " << gpuTime << "ms" << endl;
  cout << "Time elapsed on CPU: " << duration.count() << "ms" << endl;

  return 0;
} 