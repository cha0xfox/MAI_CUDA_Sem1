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

void scanLargeDeviceArray(int *output, int *input, int length, bool bcao);
void scanSmallDeviceArray(int *d_out, int *d_in, int length, bool bcao);
void scanLargeEvenDeviceArray(int *output, int *input, int length, bool bcao);

__global__ void prescan_arbitrary(int *output, int *input, int n, int powerOfTwo);
__global__ void prescan_arbitrary_unoptimized(int *output, int *input, int n, int powerOfTwo);

__global__ void prescan_large(int *output, int *input, int n, int* sums);
__global__ void prescan_large_unoptimized(int *output, int *input, int n, int *sums);

__global__ void add(int *output, int length, int *n1);
__global__ void add(int *output, int length, int *n1, int *n2);

bool isPowerOfTwo(int x);
int nextPowerOfTwo(int x);

int THREADS_PER_BLOCK = 512;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
// Оптимизация конфликта банков
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)

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
// ============================ LOGIC FUNCS

void scanLargeDeviceArray(int *d_out, int *d_in, int length, bool bcao) {
	int remainder = length % (ELEMENTS_PER_BLOCK);
	if (remainder == 0) {
		scanLargeEvenDeviceArray(d_out, d_in, length, bcao);
	}
	else {
		// Делаем скан по всему массиву
		int lengthMultiple = length - remainder;
		scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple, bcao);

		// Скан по оставшемуся массиву (более меньшему), сохраняя последнее число.
		int *startOfOutputArray = &(d_out[lengthMultiple]);
		scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder, bcao);

		add<<<1, remainder>>>(startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
	}
}

void scanSmallDeviceArray(int *d_out, int *d_in, int length, bool bcao) {
	int powerOfTwo = nextPowerOfTwo(length);

	if (bcao) {
		prescan_arbitrary<<<1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int)>>>(d_out, d_in, length, powerOfTwo);
	}
	else {
		prescan_arbitrary_unoptimized<<<1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int)>>>(d_out, d_in, length, powerOfTwo);
	}
}

void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length, bool bcao) {
	const int blocks = length / ELEMENTS_PER_BLOCK;
	const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

	int *d_sums, *d_incr;
	cudaMalloc((void **)&d_sums, blocks * sizeof(int));
	cudaMalloc((void **)&d_incr, blocks * sizeof(int));

	if (bcao) {
		prescan_large<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
	}
	else {
		prescan_large_unoptimized<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
	}

	const int sumsArrThreadsNeeded = (blocks + 1) / 2;
	if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
		// Делаем скан по всему массиву
		scanLargeDeviceArray(d_incr, d_sums, blocks, bcao);
	}
	else {
		// Нужен только один блок на скан всего массива
		scanSmallDeviceArray(d_incr, d_sums, blocks, bcao);
	}

	add<<<blocks, ELEMENTS_PER_BLOCK>>>(d_out, ELEMENTS_PER_BLOCK, d_incr);

	cudaFree(d_sums);
	cudaFree(d_incr);
}

// ============================ KERNEL FUNCS

__global__ void prescan_arbitrary(int *output, int *input, int n, int powerOfTwo){
	extern __shared__ int temp[];// allocated on invocation
	int threadID = threadIdx.x;

	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);


	if (threadID < n) {
		temp[ai + bankOffsetA] = input[ai];
		temp[bi + bankOffsetB] = input[bi];
	}
	else {
		temp[ai + bankOffsetA] = 0;
		temp[bi + bankOffsetB] = 0;
	}


	int offset = 1;
	for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // строим дерево
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadID == 0) {
		temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0; // очищаем последний элемент
	}

	for (int d = 1; d < powerOfTwo; d *= 2) // Проходим по дереву вниз
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if (threadID < n) {
		output[ai] = temp[ai + bankOffsetA];
		output[bi] = temp[bi + bankOffsetB];
	}
}

__global__ void prescan_arbitrary_unoptimized(int *output, int *input, int n, int powerOfTwo) {
	extern __shared__ int temp[];// allocated on invocation
	int threadID = threadIdx.x;

	if (threadID < n) {
		temp[2 * threadID] = input[2 * threadID]; // загружаем массив в память
		temp[2 * threadID + 1] = input[2 * threadID + 1];
	}
	else {
		temp[2 * threadID] = 0;
		temp[2 * threadID + 1] = 0;
	}


	int offset = 1;
	for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // строим дерево
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadID == 0) { temp[powerOfTwo - 1] = 0; } // очищаем последний элемент

	for (int d = 1; d < powerOfTwo; d *= 2) // Проходим по дереву вниз
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if (threadID < n) {
		output[2 * threadID] = temp[2 * threadID]; // Записываем в память устройства
		output[2 * threadID + 1] = temp[2 * threadID + 1];
	}
}


__global__ void prescan_large(int *output, int *input, int n, int *sums) {
	extern __shared__ int temp[];

	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * n;

	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = input[blockOffset + ai];
	temp[bi + bankOffsetB] = input[blockOffset + bi];

	int offset = 1;
	for (int d = n >> 1; d > 0; d >>= 1) // строим дерево
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	__syncthreads();


	if (threadID == 0) {
		sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
	}

	for (int d = 1; d < n; d *= 2) // Проходим по дереву вниз
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	output[blockOffset + ai] = temp[ai + bankOffsetA];
	output[blockOffset + bi] = temp[bi + bankOffsetB];
}

__global__ void prescan_large_unoptimized(int *output, int *input, int n, int *sums) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * n;

	extern __shared__ int temp[];
	temp[2 * threadID] = input[blockOffset + (2 * threadID)];
	temp[2 * threadID + 1] = input[blockOffset + (2 * threadID) + 1];

	int offset = 1;
	for (int d = n >> 1; d > 0; d >>= 1) // строим дерево
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	__syncthreads();


	if (threadID == 0) {
		sums[blockID] = temp[n - 1];
		temp[n - 1] = 0;
	}

	for (int d = 1; d < n; d *= 2) // Проходим по дереву вниз
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	output[blockOffset + (2 * threadID)] = temp[2 * threadID];
	output[blockOffset + (2 * threadID) + 1] = temp[2 * threadID + 1];
}


__global__ void add(int *output, int length, int *n) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n[blockID];
}

__global__ void add(int *output, int length, int *n1, int *n2) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}

// ============================
// Является ли степенью двойки
bool isPowerOfTwo(int x) {
	return x && !(x & (x - 1));
}

// Следующая степень двойки
int nextPowerOfTwo(int x) {
	int power = 1;
	while (power < x) {
		power *= 2;
	}
	return power;
}

// ============================
// C++11 way генерация случайных чисел с нормальным распределением

void testCPU(int* output, int* input, int length) {
	output[0] = 0; // первый элемент 0, так как это по сути прескан
	for (int j = 1; j < length; ++j)
	{
		output[j] = input[j - 1] + output[j - 1];
	}
}

int main() {

  using clock = std::chrono::system_clock;
  using sec = std::chrono::duration<double, std::milli>;

  constexpr int N = 10000000;

  int length = N;
  bool bcao = false;

  vector<int> h_a(N);
  vector<int> h_b(N);
  generate(h_a.begin(), h_a.end(), []() { return rand() % 10; });

  int *d_a, *d_b;

  CHECK_CUDA_ERROR(cudaMalloc(&d_a, N * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_b, N * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a.data(), N * sizeof(int), cudaMemcpyHostToDevice));

  // Создаем эвенты, которые рассчитают время выполнения на GPU.
  cudaEvent_t start,stop;
  float gpuTime = 0.0f;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  CHECK_CUDA_ERROR(cudaEventRecord(start,0));

  // Запускаем в работу GPU асинхронно
  if (length > ELEMENTS_PER_BLOCK) {
		scanLargeDeviceArray(d_b, d_a, length, bcao);
	}
	else {
		scanSmallDeviceArray(d_b, d_a, length, bcao);
	}

  CHECK_CUDA_ERROR(cudaEventRecord(stop,0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpuTime, start, stop));
  CHECK_CUDA_ERROR(cudaMemcpy(h_b.data(), d_b, N*sizeof(int), cudaMemcpyDeviceToHost));

  // Уничтожаем созданные эвенты
  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));

  CHECK_LAST_CUDA_ERROR();
  // ----------------------------
  
  cout << "GPU array: ";
  for (int i = N; i > N - 10; i--){
    cout << "[" << h_b[i] << "] ";
  }
  cout << endl;
  
  // Тестируем время выполнения на CPU
  const auto before = clock::now();

  testCPU(h_b.data(),h_a.data(),N);

  const sec duration = clock::now() - before;

  
  cout << "CPU array: ";
  for (int i = N; i > N - 10; i--){
    cout << "[" << h_b[i] << "] ";
  }
  cout << endl;
  
  // Вывод времени
  cout << "Time elapsed on GPU: " << gpuTime << "ms" << endl;
  cout << "Time elapsed on CPU: " << duration.count() << "ms" << endl;
  
  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));

  return 0;
} 