#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

__global__ void printString(int N) {
  // Глобольный тред id
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Не используем треды которые нам не нужны
  if (tid < N) printf("This is GPU example string\n");
}

void testCpu(int N) {
  for (int i = 0; i<N; i++) {
    printf("This is CPU example string\n");
  };
}

int main() {

  using clock = std::chrono::system_clock;
  using sec = std::chrono::duration<double, std::milli>;

  // N - в текущем случае количество строк
  constexpr int N = 10;
  //constexpr size_t bytes = sizeof(int) * N;

  // Количество тредов на блок (1024)
  int NUM_THREADS = 1 << 10;

  // Рассчет минимально необходимого количества блоков.
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  // Создаем эвенты, которые рассчитают время выполнения на GPU.
  cudaEvent_t start,stop;
  float gpuTime = 0.0f;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start,0);

  // Запускаем в работу GPU асинхронно
  printString<<<NUM_BLOCKS, NUM_THREADS>>>(N);

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpuTime, start, stop);

  // Уничтожаем созданные эвенты
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // ----------------------------

  // Тестируем время выполнения на CPU

  const auto before = clock::now();

  testCpu(N);

  const sec duration = clock::now() - before;

  // Выведем время

  std::cout << "Time elapsed on GPU: " << gpuTime << "ms" << std::endl;
  std::cout << "Time elapsed on CPU: " << duration.count() << "ms" << std::endl;

  return 0;
}

