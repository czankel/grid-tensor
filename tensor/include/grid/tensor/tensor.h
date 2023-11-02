//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_TENSOR_H
#define GRID_TENSOR_TENSOR_H

#include <concepts>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>

#include <grid/tensor/tensor_traits.h>

namespace grid {

/// Tensor is an implementation of an "AI Tensor" for vector and matrix operations, and follows
/// less the mathematical or physical definition.
///
/// @tparam T       Integral type
/// @tparam Rank    Rank of the tensor with 0: scalar, 1: vector, 2: matrix, etc.
/// @tparam Device  Device for
/// @tparam Memory  Underlying memory
///
/// The Tensor defines these types:
///   value_type      T*
///   device_type     underlying device .. accelerator ...
///   memory_type     memory-mapped, device, view, device view
///   pointer_type    T* or memory specific, which would require a 'map/unmap'
///   rank
///
/// The Tensor must provide the following methods for returning the dimensions and strides:
///   std::array<size_t, Rank>   Dimensions()
///   std::array<ssize_t, Rank>  Strides()
///
/// The Tensor must also either have a direct pointer to the date:
///   pointer_type Data()    // FIXME: might be hidden? Maybe use Pointer
///
/// Or, alternative, provide means to map and unmap the memory (with a penality):
///   pointer_type Map()
///   void         UnMap()
///
/// A tensor view is also a tensor but doesn't own the data buffer. Instead, it points to
/// the data buffer of another tensor. Because of the object lifetime of the tensor, it has
/// some restrictions:
///
///  - A view can only be assigned to a tensor (or tensor view).
///  - If a tensor (or tensor view is assigned to the view, the data is always copied to the
///    'view' area, even if the other object is a rvalue reference.
///

constexpr static size_t kMaxRank = 4;

// Memory Management??
//
// Could be Allocator // Allocatable 
//  - StandardAllocator / MemoryAllocator / StandardMemory / SystemMemory / HeapAllocation
// Allocator but requires map/unmap?
//  - CUDA / GPUMemory / AcceleratorMemory
//  - Metal / AcceleratoeMemory
// Mapping or Backed-By
//  - StaticAllocator / StaticMemory / Readonly / ...
//  - TensorView / MemoryView / MemoryReference / ??
//  - MemoryMapped / FileMapped / Filebackend / ...
// GDS / Remote / ...
//  - 
//
// System Memory:
//   - allocatable
//   - provide size, maybe alignment, type?
// Device Memory
//   - allocatable
//   - provide size, maybe alignment, etc.
//   - might require mapping or even copy
// FileMap
//   - fixed size, already "allocated"
//   - mapping
// Static
//   - fixed size, maybe even RO
//
// Default/Std:   SystemMemory/StandardAllocator/HeapAllocator/..Allocation
// Static/Stack?: StaticMemory/StaticAllocator/StackAllocator??
// MMap           MemoryMap/MemoryMapped/FileMapped/FileAllocator
// View           ViewMemory/View/TensorView/MemoryReference/TensorReference/Tensor
// CUDA/Metal     DeviceMemory/CUDAMemory/...
// GDS            DirectStorage/ .. .(requires one of the other memory devices for 'cacching'?)
//
/ Memory Functions
//  - Data                get pointer to underlying data
//  - Map/Unmap           map and unmap memory area (might be a copy)
//  - Allocate/Deallocate (extending doesn't really make sense? so, only initial allocation, why at all then?)
//  - ..
//
// MemoryHandler/Manager/Class/???
//  * MemoryAllocator                                       --> PROVIDE SIZE          --> Allocate
//  * Constants                                             --> PROVIDE ELEMENTS      --> Copy/Move
//  * MappeFile                                             --> PROVIDE MMAPed AREA   --> Set pointer
//  * TensorView                                            --> PROVIDE TENSOR        --> Set tensor reference
//  * CUDA / Metal / ...                                    --> PROVIDE SIZE          --> Allocate device memory
//  * DirectStorage                                         --> PROVIDE ??            --> ??

//             MemAlloc  Const    MMap    Device   GDS
// Allocate       *        -       -         *       -
// Map/Unmap      -        -       *         *       *
// Writable       *        -       *         *       *
// Cahced         -        -       *         *       *
//
// View is not a MemoryClass, it "inherits" the memory class from the Tensor
// How to differentiate View from Tensor?
//  - Different class? TensorView (...)  --> Tensor needs to have the "device" embedded, e.g. as typedef, or as template
//  - Different MemoryClass (one View per each?)
//
// Device is not really needed for Tensor? That's something for the Operators. The Tensor is only about memory,
// which does "embed" the device to some extent?
// The actual device comes in only through the Operators (including assignment!!??)
// An Operator for an accelerator might support MemoyrAllocator Tensor, it would have to implicitly convert it
// to a Device memory class. Better to have the user do it?
// For MMap, have a DeviceMMap? or have a Tensor<DeviceMemory>(Tensor<MMap>&&) ?
//
// Original idea was to just alias the Tensor type:
//
// using Tensor = TensorCPU;
// using Tensor = TensorCUDA;
//
// And, Tensor t{1,2}; or Tensor t(5UL, 4.0); would suffice.
//
// This would not work anymore reducing TensorXXX to just Tensor!?
//
// Allocate: Use different allocators (SystemMemroy or DeviceMemory)
// Const: is just a const...
//
// using Tensor2D = Tensor<T,2, <ALLOCATOR>)
// using Tensor2DConst = ...;
// using Tensor2DFile = ...;
//
// MMap -> Can be used as is for CPU but needs to be "copied" to TensorDevice!?
//
// So, different set of Tensors?
//
// TensorCPU: Allocate, Const, MMap
// TensorCUDA: Allocate, MMap (Const -> Allocate)
// TensorXYZ: ???
//
// Allocate, AllocateMMap, AllocateConst??
//
// Tensor<ALLOCATE<MEMORY>>
// Tensor<ALLOCATE<MMAP>> or Tensor<ALLOCATE<FILE>> or Tensor<MMAP<MEMORY>> vs Tensor<MMAP<CUDA>>
// Tensor<CONST<MEMORY>>
//
// Tensor<ALLOCATE<CUDA>>
// Tensor<MMAP<CUDA>>
// Tensor<CONST<CUDA>>
//
// template<Device> class MemoryAllocator;
// template<Device> class MMapAllocator;
// template<Device> class ConstAllocator;
//
// template<Allocator<Device>> class Tensor;
//
// Specializations:
//  - Tensor<MemoryAllocator<CPU>>
//  - Tensor<MMapAllocator<CPU>>
//  - Tensor<ConstAllocator<CPU, size_t...>>
//
// Device Names
//  - CPU, DefaultCPU, NoDevice, Standard, BaseLine, Compiler, DefaultOptimized
//
// template<typename Device>
// SomeFunc()
// {
//   using Tensor = Tensor<MemoryAllocator<Device>>;
//   using TensorFile = Tensor<MMapAllocator<Device>>;
//   using TensorConst = Tensor<ConstAllocator<Device>>; !!!!
//
//   using Tensor = TensorSlowCpu<T,Rank>
//   using TensorF = TensorSlowCpu<T,Rank,MMap>
//
//   using Tensor =  Tensor<T,Rank,Device>
//                   Tensor<T,Rank,MemoryAllocation<StandardDevice>>
//   using TensorF = Tensor<T,Rank,Device,MMap>
//                   Tensor<T,Rank,MMap<Device>>
//  TensorF should not be required? It's deduced...
//
//   using Tensor = grid::Tensor<T,Rank,System>; // NOT SURE THIS"LL WORK?
//   Tensor mmap = Tensor(mmap, {x,y..});
//
//  System, CUDA, Metal, ...
//    System --> Allocate, Constant, FileMemoryMap, etc.
//
//  How will this work? Tensor<T,Rank,Device> --> Tensor<T,Rank,XYZ<Device>>
//  Or,  Tensor<T,Rank,System,Allocator=StdAllocator>;
//       Tensor<T,Rank,CUDA,Allocator=CUDAAllocator>;
//
//  template <typename Device> class StandardAllocator;
//  template <> class StandardAllocator<System> {};
//  template <> class StandardAllocator<CUDA> {};
//
//  template <typename Device> class FileMemoryMap;
//  template <> class FileMemoryMap<System> {};
//  template <> class FileMemoryMap<CUDA> {};
//
//  Without CTAD
//  template<typename T, typename DEVICE>
//  struct {
//
//  using TensorFile2D = Tensor<T, 2, File<DEVICE>>;
//
//  template <typename T, size_t Rank, typename Allocator>
//  struct TensorCUDA
//  {
//    ...
//
//
// using TensorFile = Tensor<C
//
// Requirements
//
// - Want to easily switch between devices...
//
//
// 1. CTAD rules need to be in the scope of the class
// 2. CTAD rules cannot have extra template parameters
// 3. "Device" is for computation so shouldn't be part of Tensor
// 4. Allocation could support different "memory" types, but CTAD wouldn't know which
// 5. Views are different from tensors but should act as tensors; they don't have a different allocation only
//
// -> Using TensorDEVICE as scope, one would not want to match; DEVICE could use "using TensorDEVICE = TensorSystem"?
// -> Alternative, could use namespaces:  grid::tensor::system or grid::tensor::cuda
//
//
// PROPOSAL/DECISION
//
//  1. Have a 'default' set of Tensors. Name?  TensorBase, TensorSystem, Tensor?
//  2. Provide an "Allocator" template parameter, defaulting to std::allocator
//      - Default tensor implements a buffer using the provided allocator (defaults to std::allocator)
//      - Additional Allocator orverrides for special cases:
//         o Constant/ConstantAllocator/...   -> in .text section
//         o Shared memory buffer (including mmap, etc.) inlcudes a std::shared_ptr
//         o Reference/View -> special type, has assignment/copy/move restrictions
//  3. Provide separate "classes" of Tensors for different "devices"
//      - Allows users to simply "switch" between different devices using,
//        e.g. using Tensor = TensorDEVICE or Tensor = DEVICE::Tensor  << OPEN ITEM
//
// Questions
// 
//  - Would it work to have just all the base Tensors and then "group" them?
//    for each devie:
//      namespace DEVICE { using Tensor<> = ::base::Tensor<DEVICEAllocator>; ... }
//    to select
//
//      template<Allocator>
//      using Tensor = DEVICE::Tensor<Allocator>
//
//

/// TensorView is used as a non-type template parameter declaring a tensor view type.
/// This is only used internally.
struct TensorView {}; // FIXME: MemoryView

/// StandardAllocator uses the default std::allocator (new[]/delete[]) for allocating the data buffer.
//struct StandardAllocator {};

/// StaticAllocator is a ... for static allocation, meaning FIXME   rename to ConstAllocator?
///
template <size_t...> struct StaticAllocator {};

/// Broadcast defines to set the dimension to 1 ("broadcastable") in the axes argument of Tensor::View.
inline constexpr ssize_t Broadcast = -1;

/// Placeholder for specifying that a buffer allocation does not need to be initialized.
template <typename T> struct Uninitialized { using type = T; };


// Concepts

/// TensorFor<DEVICE> requires that the provided argument is a Tensor for the specific DEVICE.
template <typename _Tensor, template <typename, size_t, typename> typename _DeviceTensor>
concept TensorFor = is_tensor_v<_Tensor> && is_same_device_v<_Tensor, _DeviceTensor>;

/// AnyTensor requires that the provided argument is a Tensor
template <typename _Tensor>
concept AnyTensor = is_tensor_v<_Tensor>;


/// TensorOpFor<DEVICE> requires that the provided argument is a TensorOp for the specific DEVICE.
template <typename _TensorOp, template <typename, size_t, typename> typename _DeviceTensor>
concept TensorOpFor = is_tensor_op_v<_TensorOp> && is_same_device_v<_TensorOp, _DeviceTensor>;

/// AnyTensorOp requires that the provided argument is a TensorOp
template <typename _TensorOp>
concept AnyTensorOp = is_tensor_op_v<_TensorOp>;

/// ConvertibleTensorFor<DEVICE> requires that the provided argument can be converted to a Tensor
template <typename _Tensor, template <typename, size_t, typename> typename _DeviceTensor>
concept ConvertibleTensorFor = (is_tensor_v<_Tensor> || is_tensor_op_v<_Tensor>) && is_same_device_v<_Tensor, _DeviceTensor>;

/// AnyConvertibleTensor requires that the provided argument can be converted to a Tensor.
template <typename _Tensor>
concept AnyConvertibleTensor = is_tensor_v<_Tensor> || is_tensor_op_v<_Tensor>;

/// TensorRank<RANK> requires that the provided argument is a tensor of the rank RANK.
template <typename _Tensor, size_t _Rank>
concept TensorRank = (is_tensor_v<_Tensor> || is_tensor_op_v<_Tensor>) && _Tensor::Rank() == _Rank;

/// TensorNotRank<RANK> requires that the provided argument is a tensor that is not of the rank RANK.
template <typename _Tensor, size_t _Rank>
concept TensorNotRank = (is_tensor_v<_Tensor> || is_tensor_op_v<_Tensor>) && _Tensor::Rank() != _Rank;

// Viewable requires that a tensor can be made "viewable", which means it is not a view itself.
template <typename _Tensor>
concept Viewable = requires (const _Tensor& t) { t.View; };

/// TensorViewFor<DEVICE> requires that the provided argument is a tensor view and for a specific device
template <typename _Tensor, template <typename, size_t, typename> typename _DeviceTensor>
concept TensorViewFor = !Viewable<_Tensor> && is_same_device_v<_Tensor, _DeviceTensor>;


// Supported basic arithmetic operations for all Tensor implementations.

template <template <typename, size_t, typename> typename, typename, size_t, typename... > struct TensorAdd;
template <template <typename, size_t, typename> typename, typename, size_t, typename... > struct TensorMul;
template <template <typename, size_t, typename> typename, typename, size_t, typename... > struct TensorRmsNorm;


// Operator overloading

// operator+ (TensorType, TensorType)
template <AnyConvertibleTensor _Tensor1, AnyConvertibleTensor _Tensor2>
auto operator+(_Tensor1&& tensor1, _Tensor2&& tensor2)
{
  return TensorAdd(std::forward<_Tensor1>(tensor1), std::forward<_Tensor2>(tensor2));
}

// operator* (TensorType, TensorType)
template <AnyConvertibleTensor _Tensor1, AnyConvertibleTensor _Tensor2>
auto operator*(_Tensor1&& tensor1, _Tensor2&& tensor2)
{
  return TensorMul(std::forward<_Tensor1>(tensor1), std::forward<_Tensor2>(tensor2));
}


// helper function to get an array from a brace-initializer list.
template <typename _T, size_t... _Ns>
inline constexpr std::array<_T, sizeof...(_Ns)>
get_array_impl(std::initializer_list<_T>&& init, std::index_sequence<_Ns...>)
{
  return std::array<_T, sizeof...(_Ns)>{ *(init.begin() + _Ns) ... };
}

template <typename _T, size_t _N, typename _Ns = std::make_index_sequence<_N>>
inline constexpr std::array<_T, _N>
get_array(std::initializer_list<_T>&& init)
{
  return get_array_impl(std::move(init), _Ns{});
}

// helper function to return an array from a two-dimensional initializer list
template <typename _T, size_t _M, size_t _N>
inline constexpr std::array<_T, _M * _N>
get_array(std::initializer_list<std::initializer_list<_T>>&& init)
{
  std::array<_T, _M * _N> arr{};
  auto line_it = arr.begin();
  for (auto it : init)
  {
    std::copy(it.begin(), it.end(), line_it);
    line_it += _N;
  }
  return arr;
}

// helper function to return an array from a three-dimensional initializer list
template <typename _T, size_t _C, size_t _M, size_t _N>
inline constexpr std::array<_T, _C * _M * _N>
get_array(std::initializer_list<std::initializer_list<std::initializer_list<_T>>>&& init)
{
  std::array<_T, _C * _M * _N> arr{};
  auto line_it = arr.begin();
  for (auto lt : init)
  {
    for (auto it : lt)
    {
      std::copy(it.begin(), it.end(), line_it);
      line_it += _N;
    }
  }
  return arr;
}

// helper function to re turn an array from a c-array.
template <typename _T, size_t _N>
inline constexpr std::array<_T, _N>
get_array(const _T(&init)[_N])
{
  std::array<_T, _N> arr{};
  std::copy(std::begin(init), std::end(init), arr.begin());
  return arr;
}

// helper function to re turn an array from an rvalue c-array.
template <typename _T, size_t _N>
inline constexpr std::array<_T, _N>
get_array(_T(&&init)[_N])
{
  std::array<_T, _N> arr{};
  std::copy(std::begin(init), std::end(init), arr.begin());
  return arr;
}

// helper function to return the strides from dimensions. Use: make_strides<TYPE>(std::array)
template <typename _T, size_t _Rank, size_t... Is>
constexpr std::array<ssize_t, _Rank>
make_strides_impl(const std::array<size_t, _Rank>& dims, std::index_sequence<Is...>)
{
  auto multiply = [&dims](size_t index) {
    ssize_t res = sizeof(_T);
    for (size_t i = 0; i < _Rank - 1 - index; i++)
      res *= dims[_Rank - 1 - i];
    return res;
  };
  return std::array<ssize_t, _Rank>{multiply(Is)...};
}

template <typename _T, size_t _Rank, typename Indices = std::make_index_sequence<_Rank>>
std::array<ssize_t, _Rank> make_strides(const std::array<size_t, _Rank>& dims)
{
  return make_strides_impl<_T>(dims, Indices{});
}

/// operator<< outputs the tensor buffer.
std::ostream& operator<<(std::ostream& os, const grid::AnyTensor auto& tensor)
{
  using value_type = typename std::remove_reference_t<decltype(tensor)>::value_type;
  size_t rank = tensor.Rank();

  auto dims = tensor.Dimensions();
  auto strides = tensor.Strides();

  std::function<void(int, const value_type*&)> print;
  print = [&os, &dims, &strides, &print, &rank](size_t index, const value_type* ptr) {
    os << "{ ";
    if (index < rank -1)
    {
      for (size_t i = dims[index]; i > 0; i--)
      {
        print(index + 1, ptr);
        if (i != 1)
          os << ", ";
        else
          os << " }";
        ptr += strides[index] / sizeof(*ptr);
      }
    }
    else
    {
      auto* p = ptr;
      for (size_t i = dims[index]; i > 0; i--)
      {
        os << *p;
        if (i != 1)
          os << ", ";
        else
          os << " }";
        p += strides[index] / sizeof(*ptr);
      }
    }
  };

  const value_type* ptr = reinterpret_cast<const value_type*>(tensor.Data());
  if (rank > 0)
    print(0, ptr);
  else
    os << "{ " << *ptr << " }";

  os << std::flush;

  return os;
}

} // end of namespace grid

#endif  // GRID_TENSOR_TENSOR_H
