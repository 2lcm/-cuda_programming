#include "MatAdd_LargeMatrix.h"

/******************************************************************
* Complete following three kernels
******************************************************************/

__global__ void MatAdd_G2D_B2D
(float* MatA, float* MatB, float* MatC, int nRow, int nCol)
{
	// Write your 2D_2D kernel here
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int index = row * nCol + col;

	if (col < nCol && row < nRow)
		MatC[index] = MatA[index] + MatB[index];
}

__global__ void MatAdd_G1D_B1D
(float* MatA, float* MatB, float* MatC, int nRow, int nCol)
{
	// Write your 1D_1D kernel here
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

	if (col < nCol){
		for (unsigned int row = 0; row < nRow; row++){
			unsigned int index = row * nCol + col;
			MatC[index] = MatA[index] + MatB[index];
		}	
	}
		
}

__global__ void MatAdd_G2D_B1D
(float* MatA, float* MatB, float* MatC, int nRow, int nCol)
{
	// Write your 2D_1D kernel here
	unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int row = blockIdx.y;
	unsigned int index = row * nCol + col;

	if (col < nCol && row < nRow)
		MatC[index] = MatA[index] + MatB[index];
}




bool kernelCall(float* _MatA, float* _MatB, float* _MatC, int _nRow, int _nCol
	, int _layout, dim3 _gridDim, dim3 _blockDim)
{
	switch (_layout)
	{
	case ThreadLayout::G1D_B1D:
		MatAdd_G1D_B1D <<<_gridDim, _blockDim >>> (_MatA, _MatB, _MatC, _nRow, _nCol);
		break;
	case ThreadLayout::G2D_B1D:
		MatAdd_G2D_B1D <<<_gridDim, _blockDim >>> (_MatA, _MatB, _MatC, _nRow, _nCol);
		break;
	case ThreadLayout::G2D_B2D:
		MatAdd_G2D_B2D <<<_gridDim, _blockDim >>> (_MatA, _MatB, _MatC, _nRow, _nCol);
		break;
	default:
		printf("Not supported layout\n");
		return false;
	}
	return true;
}