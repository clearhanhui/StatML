[TOC]

# 稀疏矩阵

插播一期稀疏矩阵。

### 为什么稀疏矩阵
在实际应用中，矩阵大多时候都是稀疏的（例如大图的邻接矩阵），稀疏矩阵能减少存储空间，加快计算速度。

### 常用稀疏矩阵
##### 1. coo：Coordinate matrix

采用三个数组，row，col，data，分别表示 行坐标，列坐标，和该坐标系下对应的值。下面的例子是用scipy.sparse创建coo稀疏矩阵。

``` python 
>>> from scipy.sparse import coo_matrix
>>> row  = np.array([0, 3, 1, 0])
>>> col  = np.array([0, 3, 1, 2])
>>> data = np.array([4, 5, 7, 9])
>>> coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
array([[4, 0, 9, 0],
       [0, 7, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 5]])
```

**优点**：

* 方便稀疏格式之间的快速转换；

* 允许重复条目（参见示例）；

* 与 CSR/CSC 格式之间的快速转换； 

**缺点** ：

* 不支持数学运算；
* 不支持切片（slice）。

##### 2. csr和csc：Compressed Sparse Row/Column matrix

分别代表按行和按列的压缩方式。下面只介绍csr，csc和csr类似。

采用三个数组，data，indices，indptr，分别表示 数值，列号，和偏移量。对应的稠密矩阵的第$i$ 行的数据表示为（python）：
``` python
for i in range(len(indptr)-1):
    for j in range(indptr[i],indptr[i+1]):
		matrix[i][indices[j]] = data[j]
```
下面用scipy.sparse创建csr稀疏矩阵的例子：
```
>>> import numpy as np
>>> from scipy.sparse import csr_matrix
>>> indptr = np.array([0, 2, 3, 6])
>>> indices = np.array([0, 2, 2, 0, 1, 2])
>>> data = np.array([1, 2, 3, 4, 5, 6])
>>> csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
array([[1, 0, 2],
       [0, 0, 3],
       [4, 5, 6]])
```

**优点**：

* 高效算术运算 CSR + CSR、CSR * CSR 等；

* 高效的行切片（row slice）；

* 快速矩阵向量乘积 ；

**缺点**：

* 缓慢的列切片（column slice）操作（考虑使用CSC）；
* 稀疏结构的改变代价高昂（考虑 LIL 或 DOK）。

> All conversions among the CSR, CSC, and COO formats are efficient, linear-time operations.

### python中的[scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html#module-scipy.sparse)

##### 支持的稀疏格式

| 矩阵格式                                                     | 描述                                                      |
| ------------------------------------------------------------ | --------------------------------------------------------- |
| [`bsr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html#scipy.sparse.bsr_matrix)(arg1[, shape, dtype, copy, blocksize]) | Block Sparse Row matrix                                   |
| [`coo_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix)(arg1[, shape, dtype, copy]) | A sparse matrix in COOrdinate format.                     |
| [`csc_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html#scipy.sparse.csc_matrix)(arg1[, shape, dtype, copy]) | Compressed Sparse Column matrix                           |
| [`csr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix)(arg1[, shape, dtype, copy]) | Compressed Sparse Row matrix                              |
| [`dia_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.html#scipy.sparse.dia_matrix)(arg1[, shape, dtype, copy]) | Sparse matrix with DIAgonal storage                       |
| [`dok_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.html#scipy.sparse.dok_matrix)(arg1[, shape, dtype, copy]) | Dictionary Of Keys based sparse matrix.                   |
| [`lil_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html#scipy.sparse.lil_matrix)(arg1[, shape, dtype, copy]) | Row-based list of lists sparse matrix                     |
| [`spmatrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.spmatrix.html#scipy.sparse.spmatrix)([maxprint]) | This class provides a base class for all sparse matrices. |

##### 常用api

| API                                                          | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`eye`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.eye.html#scipy.sparse.eye)(m[, n, k, dtype, format]) | Sparse matrix with ones on diagonal                          |
| [`identity`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.identity.html#scipy.sparse.identity)(n[, dtype, format]) | Identity matrix in sparse format                             |
| [`hstack`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.hstack.html#scipy.sparse.hstack)(blocks[, format, dtype]) | Stack sparse matrices horizontally (column wise)             |
| [`vstack`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.vstack.html#scipy.sparse.vstack)(blocks[, format, dtype]) | Stack sparse matrices vertically (row wise)                  |
| [`random`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.random.html#scipy.sparse.random)(m, n[, density, format, dtype, …]) | Generate a sparse matrix of the given shape and density with randomly distributed values. |
|                                                              |                                                              |
| [`save_npz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.save_npz.html#scipy.sparse.save_npz)(file, matrix[, compressed]) | Save a sparse matrix to a file using `.npz` format.          |
| [`load_npz`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.load_npz.html#scipy.sparse.load_npz)(file) | Load a sparse matrix from a file using `.npz` format.        |
|                                                              |                                                              |
| [`multiply`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.multiply.html#scipy.sparse.coo_matrix.multiply)(other) | Point-wise multiplication by another matrix                  |
| [`power`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.power.html#scipy.sparse.coo_matrix.power)(n[, dtype]) | This function performs element-wise power.                   |
| [`dot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.dot.html#scipy.sparse.coo_matrix.dot)(other) | Ordinary dot product                                         |

