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
using std::endl;

__device__ long ackermannCuda(long m, long n) {
    if (!m) return n+1;
    if (!n) return ackermannCuda(m-1,1);
    return ackermannCuda(m-1, ackermannCuda(m, n-1));
}

__global__ void ackermannCudaMain(long m, long n) {
    int tid_m = (blockIdx.x * blockDim.x) + threadIdx.x;
    for (int n=0; n<6-tid_m; n++)
        printf("CUDA A(%d,%d) = %d\n",tid_m,n,ackermannCuda(tid_m,n));
}

int ackermann_rec(unsigned int m,unsigned int n){
    if (!m) return n+1;
    if (!n) return ackermann_rec(m-1,1);
    return ackermann_rec(m-1, ackermann_rec(m, n-1));
}

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

void testCpu() {
    int m, n;
    for (m=0; m<=4; m++)
            for (n=0; n<6-m; n++)
                    printf("A(%d, %d) = %d\n", m, n, ackermann_rec(m,n));
}


int main() {

  using clock = std::chrono::system_clock;
  using sec = std::chrono::duration<double, std::milli>;

  constexpr int N = 100;
  constexpr size_t bytes = sizeof(int) * N;

  // Количество тредов на блок
  int NUM_THREADS = 32;

  // Рассчет минимально необходимого количества блоков.
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
  // int NUM_BLOCKS = 10;

  dim3 threads(NUM_THREADS, NUM_THREADS);
  dim3 blocks(NUM_BLOCKS, NUM_BLOCKS);

  // Создаем эвенты, которые рассчитают время выполнения на GPU.
  cudaEvent_t start,stop;
  float gpuTime = 0.0f;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start,0);

  cudaDeviceSetLimit(cudaLimitStackSize, 1 << 16);

  // Запускаем в работу GPU асинхронно
  ackermannCudaMain<<<5, 1>>>(4,0);
  cout << cudaGetLastError() << endl;

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpuTime, start, stop);

  // Уничтожаем созданные эвенты
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // ----------------------------


  // Тестируем время выполнения на CPU

  const auto before = clock::now();

  //testCpu();

  const sec duration = clock::now() - before;

  // Вывод времени

  cout << "Time elapsed on GPU: " << gpuTime << "ms" << endl;
  cout << "Time elapsed on CPU: " << duration.count() << "ms" << endl;

  return 0;
} 