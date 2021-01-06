# ECE 695 CUDA Programming Part 1 <br><br> Professor Tim Rogers


## Introduction

The purpose of this lab is to familiarize yourself with CUDA programming and the fundamental methodology for parallelizing a traditional CPU-based algorithm for GPU acceleration.  

The official [CUDA Documentation](https://docs.nvidia.com/cuda/) is the best resource for  implementation details and API specifics. You will find yourself referring back to various sections of this page over the next few weeks as we delve deeper into the CUDA API and the architecture of a GP-GPU. 

## PART A: Single-precision A · X Plus Y (SAXPY)

SAXPY stands for `Single-Precision A·X Plus Y`.  It is a function in the standard [Basic Linear Algebra Subroutines (BLAS)](http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) library. SAXPY is a combination of scalar multiplication and vector addition, and it’s very simple: it takes as input two vectors of 32-bit floats X and Y with N elements each, and a scalar value A. It multiplies each element `X[i]` by `A` and adds the result to `Y[i]`.

- explain premise
- explain CPU code
- explain parallelization scheme for GPU
- mention expected input - output format
- 

## PART B: Monte Carlo estimation of the value of _&pi;_

Monte Carlo methods are a broad class of computational algorithms that rely on repeated random sampling to obtain numerical results. [[1]](#1)
Among a wide vareity of applications, this method can be utilized to estimate the value of _&pi;_ as described below.

- describe monte-carlo pi
- explain CPU code
- explain parallelization scheme
    - add reduction optimization
- mention expected input-output format

## References
<a id="1">[1]</a> 
[Wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_method)
