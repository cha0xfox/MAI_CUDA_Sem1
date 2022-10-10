#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <vector> // Для векторов
#include <algorithm> // Для generate

#include <cuda_runtime.h> 

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

__device__ long ackermannCuda(long m, long n) {
    if (!m) return n+1;
    if (!n) return ackermannCuda(m-1,1);
    return ackermannCuda(m-1, ackermannCuda(m, n-1));
}

// Основная функция, вызывает функцию ackermannCuda по количеству нужных чисел.
__global__ void ackermannCudaMain(long m, long n) {
    int tid_m = (blockIdx.x * blockDim.x) + threadIdx.x;
    int tid_n = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (tid_m < 5 && tid_n < 6 - tid_m){
        if (tid_m == 4 && tid_n > 0)
        printf("Out of stack size\n");
        else
        printf("CUDA A(%d,%d) = %d\n",tid_m,tid_n,ackermannCuda(tid_m,tid_n));
    }
}


// Реализация функции аккермана в классическом виде, с помощью рекурсии
int ackermann_rec(unsigned int m,unsigned int n){
    if (!m) return n+1;
    if (!n) return ackermann_rec(m-1,1);
    return ackermann_rec(m-1, ackermann_rec(m, n-1));
}


// Реализация функции аккерамана в виде цикла, работает много медленнее
long ackermann(long m, long n){
    vector<long> mList = {m};
    while(!mList.empty()){
        m = mList.back();
        mList.pop_back();
        if (m == 0){
            ++n;
        } else if (n == 0){
            mList.push_back(--m);
            n = 1;
        }
        else {
            mList.push_back(--m);
            mList.push_back(++m);
            --n;
        }
    }
    return n;
}

// Для вывода будем использовать все значения от 0,0 до 4,1 -> 65533
// Но для сравнения ограничимся 4,0
void testCpu() {
    int m, n;
    for (m=0; m<=4; m++)
            for (n=0; n<6-m; n++){
                if (m == 4 && n > 0)
                printf("Out of stack size\n");
                else
                printf("A(%d, %d) = %d\n", m, n, ackermann_rec(m,n));
            }
}


int main() {

  using clock = std::chrono::system_clock;
  using sec = std::chrono::duration<double, std::milli>;

  constexpr int N = 10;

  // Количество тредов на блок
  int NUM_THREADS = 32;

  // Рассчет минимально необходимого количества блоков.
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
  // int NUM_BLOCKS = 10;

  dim3 threads(NUM_THREADS, NUM_THREADS);
  dim3 blocks(NUM_BLOCKS, NUM_BLOCKS);

  size_t size;
  // В дефолтных значениях размер стака в девайсе Cuda = 1кб, 
  // для работы нашей рекурсии нужно увеличить его
  cudaDeviceGetLimit(&size, cudaLimitStackSize);
  cout << "Default StackSize: " << size << endl;
  // Это максимальное значение на которое можно увеличить размер стака на моем устройстве
  CHECK_CUDA_ERROR(cudaDeviceSetLimit(cudaLimitStackSize, 131072ULL));
  cudaDeviceGetLimit(&size, cudaLimitStackSize);
  cout << "StackSize: " << size << endl;

  // Создаем эвенты, которые рассчитают время выполнения на GPU.
  cudaEvent_t start,stop;
  float gpuTime = 0.0f;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  CHECK_CUDA_ERROR(cudaEventRecord(start,0));

  // Запускаем в работу GPU асинхронно
  ackermannCudaMain<<<blocks, threads>>>(4,0);

  CHECK_CUDA_ERROR(cudaEventRecord(stop,0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpuTime, start, stop));

  // Уничтожаем созданные эвенты
  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));

  CHECK_LAST_CUDA_ERROR();
  // ----------------------------


  // Тестируем время выполнения на CPU

  const auto before = clock::now();

  testCpu();

  const sec duration = clock::now() - before;

  // Вывод времени

  cout << "Time elapsed on GPU: " << gpuTime << "ms" << endl;
  cout << "Time elapsed on CPU: " << duration.count() << "ms" << endl;

  return 0;
} 