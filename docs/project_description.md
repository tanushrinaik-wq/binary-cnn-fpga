# Project Description

## Title

Binary Convolution Engine using XNOR and Popcount

## Objective

To implement a hardware-efficient convolution operation by replacing multiplication with XNOR and accumulation with popcount.

## Specifications

* Input: 8x8 binary image
* Kernel: 3x3 binary filter
* Output: 6x6 feature map

## Core Idea

Binary neural networks reduce computation cost by replacing arithmetic operations with bitwise logic.

## Flow

1. Extract 3x3 window from input
2. Perform XNOR with kernel
3. Count number of 1s (popcount)
4. Output result
