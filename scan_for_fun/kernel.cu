#include <cuda.h>
#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>

#ifdef __CUDACC__
#define cuda_SYNCTHREADS() __syncthreads()
#else
#define cuda_SYNCTHREADS()
#endif

const int N_thread = 1024; // число нитей на блок

// __global__ void matrixAdd(int* firstMatrix[], int* secondMatrix[], int* resultMatrix[])
// {
// 	int col = blockIdx.x * blockDim.x + threadIdx.x;
// 	int raw = blockIdx.y * blockDim.y + threadIdx.y;
// }

__global__ void vectorAdd(int* firstVec, int* secondVec, int* resVec)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	resVec[idx] = firstVec[idx] + secondVec[idx];
}


__global__ void cumSum(int* const __restrict__ inputVec, int* __restrict__ resVec) // оптимизация с const-ами и restrict-ами
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int sharedVec[N_thread];
	sharedVec[idx] = inputVec[idx];
	cuda_SYNCTHREADS();
	for (int offset = 1; offset < N_thread; offset *= 2)
	{
		if (idx >= offset)
			sharedVec[idx] += sharedVec[idx - offset];
		else
			sharedVec[idx] = sharedVec[idx];
		cuda_SYNCTHREADS();
	}
	resVec[idx] = sharedVec[idx];
}

__global__ void transposingMatrix(float* const __restrict__ inputMatrix, float* outputMatrix, int const width,
	int const height)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	if ((idx < width) && (idy < height))
	{
		//Линейный индекс элемента строки исходной матрицы  
		int inputIdx = idx + width * idy;

		//Линейный индекс элемента столбца матрицы-результата
		int outputIdx = idy + height * idx;

		outputMatrix[outputIdx] = inputMatrix[inputIdx];
	}
}

void PrintNeededInform()
{
	int dev = 0;
	int driverVersion = 0;
	int runtimeVersion = 0;
	cudaSetDevice(dev);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf(" Device %d -> %s \n", dev, deviceProp.name);

	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	printf(" CUDA Driver Version / Runtime Version %d.%d / %d.%d\n",
		driverVersion / 1000, (driverVersion % 100) / 10,
		runtimeVersion / 1000, (runtimeVersion % 100) / 10);
	printf(" CUDA Capability Major/Minor version number: %d.%d\n",
		deviceProp.major, deviceProp.minor);
	printf(" Total amount of global memory: %.2f GBytes (%llu bytes)\n",
		static_cast<float>(deviceProp.totalGlobalMem) / (pow(1024.0, 3)),
		static_cast<unsigned long long>(deviceProp.totalGlobalMem));
	printf(" GPU Clock rate: %.0f MHz (%0.2f GHz)\n",
		deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
	printf(" Memory Clock rate: %.0f Mhz\n",
		deviceProp.memoryClockRate * 1e-3f);
	printf(" Memory Bus Width: %d-bit\n",
		deviceProp.memoryBusWidth);
	if (deviceProp.l2CacheSize)
	{
		printf(" L2 Cache Size: %d bytes\n",
			deviceProp.l2CacheSize);
	}
	printf(" Max Texture Dimension Size (x,y,z) 1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
		deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
		deviceProp.maxTexture2D[1],
		deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
		deviceProp.maxTexture3D[2]);
	printf(" Max Layered Texture Size (dim) x layers 1D = (%d) x %d, 2D = (%d, %d) x %d\n",
		deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
		deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
		deviceProp.maxTexture2DLayered[2]);

	printf(" Total amount of constant memory: %lu bytes\n",
		deviceProp.totalConstMem);
	printf(" Total amount of shared memory per block: %lu bytes\n",
		deviceProp.sharedMemPerBlock);
	printf(" Total number of registers available per block: %d\n",
		deviceProp.regsPerBlock);
	printf(" Warp size: %d\n", deviceProp.warpSize);
	printf(" Maximum number of threads per multiprocessor: %d\n",
		deviceProp.maxThreadsPerMultiProcessor);
	printf(" Maximum number of threads per block: %d\n",
		deviceProp.maxThreadsPerBlock);
	printf(" Maximum sizes of each dimension of a block: %d x %d x %d\n",
		deviceProp.maxThreadsDim[0],
		deviceProp.maxThreadsDim[1],
		deviceProp.maxThreadsDim[2]);
	printf(" Maximum sizes of each dimension of a grid: %d x %d x %d\n",
		deviceProp.maxGridSize[0],
		deviceProp.maxGridSize[1],
		deviceProp.maxGridSize[2]);
	printf(" Maximum memory pitch: %lu bytes\n", deviceProp.
		memPitch);
}

bool get_value(cudaError_t errMem)
{
	if (errMem != cudaSuccess)
	{
		printf("cannot allocate GPU memory: %s\n", cudaGetErrorString(errMem));
		return true;
	}
	return false;
}

bool addVectors()
{
	const int arraySize = 6000; // размер массива
	const int N_thread = 512; // число нитей на блок
	int N_blocks = 0; // число блоков
	int* firstVec = new int[arraySize]; // массивы на хосте
	int* secondVec = new int[arraySize];
	int* resVec = new int[arraySize];
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (int i = 0; i < arraySize; ++i)
	{
		firstVec[i] = rand();
		secondVec[i] = rand();
	}
	int* pFirstVec; // массив на видеокарте
	int* pSecondVec;
	int* pResVec;

	//определяем количество блоков
	if ((arraySize % N_thread) == 0)
	{
		N_blocks = arraySize / N_thread;
	}
	else
	{
		N_blocks = static_cast<int>(arraySize / N_thread) + 1;
	}
	dim3 blocks(N_blocks);

	cudaError_t errMem1 = cudaMalloc((void**)&pFirstVec, arraySize * sizeof(int));
	if (get_value(errMem1))
	{
		return true;
	}

	cudaError_t errMem2 = cudaMalloc((void**)&pSecondVec, arraySize * sizeof(int));
	if (get_value(errMem2))
	{
		return true;
	}

	cudaError_t errMem3 = cudaMalloc((void**)&pResVec, arraySize * sizeof(int));
	if (get_value(errMem3))
	{
		return true;
	}

	cudaEventRecord(start, 0);

	cudaMemcpy(pFirstVec, firstVec, arraySize * sizeof(int), cudaMemcpyHostToDevice); //копирование на GPU
	cudaMemcpy(pSecondVec, secondVec, arraySize * sizeof(int), cudaMemcpyHostToDevice);


	vectorAdd << <N_blocks, N_thread >> > (pFirstVec, pSecondVec, pResVec); // кол-во блоков и кол-во тредов в блоке

	cudaError_t err;
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Cannot launch CUDA kernel : %s\n", cudaGetErrorString(err));
		return true;
	}

	cudaEvent_t syncEvent;
	cudaEventCreate(&syncEvent); //Создаем event
	cudaEventRecord(syncEvent, 0); //Записываем event
	cudaEventSynchronize(syncEvent); //Синхронизируем event

	cudaMemcpy(resVec, pResVec, sizeof(int) * arraySize, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	float timeLength;
	cudaEventElapsedTime(&timeLength, start, stop);
	printf("GPU calculation time: %f ms\n", timeLength);

	printf("First vector:\n");
	for (int i = 0; i < arraySize; i++)
	{
		printf(" %d", firstVec[i]);
	}
	printf("\n Second vector:\n");
	for (int i = 0; i < arraySize; i++)
	{
		printf(" %d", secondVec[i]);
	}

	printf("\nResult vector:\n");
	for (int i = 0; i < arraySize; i++)
	{
		printf(" %d", resVec[i]);
	}

	cudaEventDestroy(syncEvent);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(firstVec);
	cudaFree(secondVec);
	cudaFree(resVec);

	delete[] firstVec;
	delete[] secondVec;
	delete[] resVec;

	cudaDeviceReset();
	return false;
}

__host__ void cumSum()
{
	int arraySize = 10;
	int* firstVec = new int[arraySize]; // массивы на хосте
	for (int i = 0; i < arraySize; ++i)
	{
		firstVec[i] = i;
		printf("%i ", firstVec[i]);
	}
	printf("\n");
	int* resVec = new int[arraySize];
	int accumulateVar = 0;
	for (int i = 0; i < arraySize; i++)
	{
		accumulateVar = accumulateVar + firstVec[i];
		resVec[i] = accumulateVar;
	}

	for (int i = 0; i < arraySize; i++)
	{
		printf("%i ", resVec[i]);
	}
}

__host__ bool asyncCummulativeSumm()
{
	const int arraySize = N_thread; // размер массива

	const int N_streams = 2;
	int N_blocks = 0; // число блоков

	if ((arraySize % N_thread) == 0) //определяем количество блоков
	{
		N_blocks = arraySize / N_thread;
	}
	else
	{
		N_blocks = static_cast<int>(arraySize / N_thread) + 1;
	}
	dim3 blocks(N_blocks);
	cudaStream_t stream[N_streams];
	for (int i = 0; i < N_streams; ++i)cudaStreamCreate(&stream[i]); // создаём стримы выполнения

	unsigned int memSizeOnOneStream = arraySize * sizeof(int) / N_streams;
	int* vec = nullptr;
	cudaMallocHost((void**)&vec, 2 * memSizeOnOneStream);
	int* resVec = nullptr;
	cudaMallocHost((void**)&resVec, 2 * memSizeOnOneStream);

	for (int i = 0; i < arraySize; ++i)
	{
		vec[i] = i;
	}

	int* pVec; // массив на видеокарте
	int* pResVec;

	cudaError_t errMem1 = cudaMalloc((void**)&pVec, 2 * memSizeOnOneStream);
	if (get_value(errMem1))
	{
		return true;
	}
	cudaError_t errMem2 = cudaMalloc((void**)&pResVec, 2 * memSizeOnOneStream);
	if (get_value(errMem2))
	{
		return true;
	}

	//асинхронное копирование на GPU
	for (int i = 0; i < N_streams; i++)
	{
		cudaMemcpyAsync(pVec + i * (arraySize / N_streams),
			vec + i * (arraySize / N_streams),
			memSizeOnOneStream,
			cudaMemcpyHostToDevice, stream[i]);
	}

	for (int i = 0; i < N_streams; i++)
	{
		cumSum << <N_blocks, N_thread, 0, stream[i] >> > (pVec, pResVec); // кол-во блоков и кол-во тредов в блоке
	}

	for (int i = 0; i < N_streams; i++)
	{
		cudaMemcpyAsync(resVec + i * (arraySize / N_streams),
			pResVec + i * (arraySize / N_streams),
			memSizeOnOneStream,
			cudaMemcpyDeviceToHost,
			stream[i]);
	}

	cudaDeviceSynchronize();

	for (int i = 0; i < N_streams; i++)
	{
		cudaStreamDestroy(stream[i]);
	}

	cudaError_t err;
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Cannot launch CUDA kernel : %s\n", cudaGetErrorString(err));
		return true;
	}

	printf("Vector:\n");
	for (int i = 0; i < arraySize; i++)
	{
		printf(" %d", vec[i]);
	}

	printf("\nResult vector:\n");
	for (int i = 0; i < arraySize; i++)
	{
		printf(" %d", resVec[i]);
	}

	cudaFreeHost(vec);
	cudaFreeHost(resVec);

	cudaDeviceReset();
	return false;
}

int main(void)
{
	//	PrintNeededInform();
	//if (addVectors()) return 1;
	//cumSum();

	if (asyncCummulativeSumm()) return 1;

	system("pause");
	return 0;
}
