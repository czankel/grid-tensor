# grid-tensor

> **_NOTE:_**  This repository is still in its infancy and evolving.
> Please check back again in future.

## Introduction

Grid Tensor is a heavily C++ templated implementation of "AI Tensors". These tensors are more
about storing data and typical vector and matrix operations rather than a mathematical or
physical definition of a Tensor.

The implementation relies heavily on C++ templates and C++20 features. This provides and
abstraction layer for models that can be mostly agnostic to the underlying implementations
with optimizations for CPUs and accelerators.

```
using Tensor = TensorFastCpu...

Tensor t1{1.0, 2.0, 3.0};
Tensor t2{3.2, 4.1, 2.3};
auto res = t1 + t2 * 3;
```
(Note that the Tensor alias (using Tensor = ) requires additional template statements removed
for brevity)

Using C++ templates also has it's disadvantages when it comes to compiler errors and debugging.
More details will be provided in a developer's guide.

## Building grid-tensor

Note that Grid Tensor requires a more recent version gcc. There is also not realy any executable
yet except for unit tests.

Assuming Grid Tensor should be built in the current directory and SRC points to the git
repository, use the following commands to build the unit tests:

```
cmake $SRC -DBUILD_TEST=1
make
./gridtensor_test
```
