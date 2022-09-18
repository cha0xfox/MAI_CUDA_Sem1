#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <vector> // Для векторов
#include <algorithm> // Для generate

using std::vector;
using std::generate;
using std::cout;
using std::endl;

__global__ void matrixM(const int *a, const int *b, int *c, int N) {
  // Глобольный тред id
  int tid_row = (blockIdx.y * blockDim.y) + threadIdx.y;
  int tid_col = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Важно проверять границы используемых тредов.
  if (tid_row < N && tid_col < N) {
    c[tid_row * N + tid_col] = 0;
    for (int k = 0; k < N; k++) {
      // Считаем для одного элемента
      c[tid_row * N + tid_col] += a[tid_row * N + k] * b[k * N + tid_col];
    }
  }
}

void testCpu(vector<int> &a, vector<int> &b, vector<int> &c, int N) {
  // Строки
  for (int i = 0; i < N; i++) {
    // Столбцы
    for (int j = 0; j < N; j++) {
      // Считаем для каждого элемента
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        tmp += a[i * N + k] * b[k * N + j];
      }
      c[i*N + j] = tmp;
    }
  }
}

void printM(vector<int> &vec, int N){
  int k = 0;
  for (int i = 0; i < vec.size(); i++){
    k++;
    cout << vec[i] << " ";
    if (k == N) {
      cout << endl;
      k = 0;
    }
  }
}

int main() {

  using clock = std::chrono::system_clock;
  using sec = std::chrono::duration<double, std::milli>;

  // N - в текущем случае количество элементов в строке матрицы
  constexpr int N = 1000;
  constexpr size_t bytes = sizeof(int) * N * N;

  // Векторы матриц
  vector<int> host_a(N * N);
  vector<int> host_b(N * N);
  vector<int> host_c(N * N);
  vector<int> host_c_cpu(N * N);

  // Рандомные значения в матрице

  generate(host_a.begin(), host_a.end(), []() { return rand() % 10; });
  generate(host_b.begin(), host_b.end(), []() { return rand() % 10; });

  // Количество тредов на блок
  int NUM_THREADS = 32;

  // Рассчет минимально необходимого количества блоков.
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
  // int NUM_BLOCKS = 10;

  // Переменные для device
  int *dev_a, *dev_b, *dev_c;
  cudaMalloc(&dev_a, bytes);
  cudaMalloc(&dev_b, bytes);
  cudaMalloc(&dev_c, bytes);

  // Копируем на device
  cudaMemcpy(dev_a, host_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, host_b.data(), bytes, cudaMemcpyHostToDevice);

  dim3 threads(NUM_THREADS, NUM_THREADS);
  dim3 blocks(NUM_BLOCKS, NUM_BLOCKS);

  // Создаем эвенты, которые рассчитают время выполнения на GPU.
  cudaEvent_t start,stop;
  float gpuTime = 0.0f;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start,0);

  // Запускаем в работу GPU асинхронно
  matrixM<<<blocks, threads>>>(dev_a, dev_b, dev_c, N);

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpuTime, start, stop);

  // Уничтожаем созданные эвенты
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Копируем вектор на хост
  cudaMemcpy(host_c.data(), dev_c, bytes, cudaMemcpyDeviceToHost);

  // ----------------------------


  // Тестируем время выполнения на CPU

  const auto before = clock::now();

  testCpu(host_a, host_b, host_c_cpu, N);

  const sec duration = clock::now() - before;

  // Выводы матриц
  if (N <= 5) {
    cout << "----Matrix A----" << endl;
    printM(host_a, N);
    cout << "----Matrix B----" << endl;
    printM(host_b, N);

    cout << "----CPU Matrix----" << endl;
    printM(host_c_cpu, N);

    cout << "----GPU Matrix----" << endl;
    printM(host_c, N);
  };
  // Вывод времени

  cout << "Time elapsed on GPU: " << gpuTime << "ms" << endl;
  cout << "Time elapsed on CPU: " << duration.count() << "ms" << endl;

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return 0;
} 