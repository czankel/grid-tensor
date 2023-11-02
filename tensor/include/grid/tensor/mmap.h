//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_TENSOR_MMAP_H
#define GRID_TENSOR_MMAP_H

#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <sys/mman.h>

#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

namespace grid {

class MMapView;
template <typename, size_t> struct MMapArray;

/// MMap represents a memory-maped file.
class MMap
{
 public:
  MMap() : fd_(-1), addr_(nullptr), file_size_(0) {}

  /// Constructor for memory-mapped file specified by the file name/path.
  MMap(const std::string& name);

  /// Constructor for memory-mapped file specified by file-descriptor and memory-mapped size.
  MMap(int fd, size_t file_size)
  {
    Map(fd, file_size);
  }

  /// Copy constructor.
  MMap(const MMap& other)
  {
    Map(other.fd_, other.file_size_);
  }

  /// Move constructor.
  MMap(MMap&& other) : fd_(other.fd_), addr_(other.addr_), file_size_(other.Size())
  {
    other.addr_= nullptr;
    other.fd_ = -1;
    other.file_size_ = 0;
  }

  /// Destructor.
  ~MMap()
  {
    Close();
  }

  MMap& operator=(MMap&& other)
  {
    Close();
    fd_ = other.fd_;
    file_size_ = other.file_size_;
    addr_ = other.addr_;

    return *this;
  }


  MMap& operator=(const MMap& other)
  {
    Map(dup(other.fd_), other.file_size_);
    return *this;
  }


  // Size returns the size of the mmaped file
  size_t Size() const                                     { return file_size_; }

  // Address returns the address of the mmaped file
  void* Address() const                                   { return addr_; }

  // End of the mmaped region
  void* End() const                                       { return addr_ + file_size_; }

 private:
  void Close()
  {
    if (addr_ != nullptr && file_size_ > 0)
      munmap(addr_, file_size_);
    if (fd_ != -1)
      close(fd_);
  }

  void Map(int fd, size_t file_size)
  {
    fd_ = fd;
    addr_ = static_cast<char*>(mmap(NULL, file_size, PROT_READ, MAP_FILE | MAP_SHARED, fd, 0));
    file_size_ = file_size;

    if (addr_ == MAP_FAILED)
      throw("mmap failed");
  }

 protected:
  int     fd_;
  char*   addr_;
  size_t  file_size_;
};

// FIXME: rename to MMapAllocator or MemoryMapAllocator
// FIXME: move to mmap.h??
/// MemoryMapped is used as a non-type template parameter declaring a memory-mapped tensor type.
/// Defining a memory-mapped tensor ... TensorImplementation<double, 2, grid::MemoryMapped{})
template <typename T> class MemoryMapped
{
 public:
  using value_type = T;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;

  MemoryMapped(std::shared_ptr<MMap> mmap, size_t offset = 0UL)
    : mmap_(mmap),
      base_(static_cast<char*>(mmap->Address()) + offset),
      addr_(static_cast<char*>(mmap->Address()) + offset),
      end_(static_cast<char*>(mmap->End()) - offset)
  {}

  [[nodiscard]] constexpr T* allocate(std::size_t n)
  {
  }

  constexpr void deallocate(T* p, std::size_t n);

 private:
  std::shared_ptr<MMap> mmap_;
  char* base_;
  char* addr_;
  char* end_;
};



/// MMapView provides a "view" into a memory-mmaped file and includes a current position for
/// sequential "read" operations.
class MMapView
{
 public:
  MMapView(std::shared_ptr<MMap> mmap, size_t offset = 0UL)
    : mmap_(mmap),
      base_(static_cast<char*>(mmap->Address()) + offset),
      addr_(static_cast<char*>(mmap->Address()) + offset),
      end_(static_cast<char*>(mmap->End()) - offset)
  {}

  /// Read returns the value of the specified type at the current position and advances the position.
  /// The current position may be unaligned.
  template<typename T>
  T Read()
  {
    char* next = addr_ + sizeof(T);
    if (next > end_)
      throw std::out_of_range("mmap read: exceeding memory-mapped area");

    T temp;
    memcpy(&temp, addr_, sizeof(T));
    addr_ = next;
    return temp;
  }

  /// Read copies the data to the provided value from the current position and advances the position.
  /// The current position may be unaligned.
  template<typename T>
  void Read(T& temp)
  {
    char* next = addr_ + sizeof(T);
    if (next > end_)
      throw std::out_of_range("mmap read: exceeding memory-mapped area");

    memcpy(&temp, addr_, sizeof(T));
    addr_ = next;
  }

  /// Read copies data from the current position to the provided destination with the provided lenght.
  template<typename T>
  void Read(T* dest, size_t len)
  {
    char* next = addr_ + len;
    if (next > end_)
      throw std::out_of_range("mmap read: exceeding memory-mapped area");

    memcpy(reinterpret_cast<char*>(dest), addr_, len);
    addr_ = next;
  }

  /// ReadString returns a std::string at the current position encoded as lenght, string and
  /// advances the position.
  std::string ReadString()
  {
    char* next = addr_ + 1;
    if (next >  end_)
      throw std::out_of_range("mmap readstring: exceeding memory-mapped area");

    uint32_t len;
    memcpy(&len, addr_, sizeof(uint32_t));
    char* str = next;
    next += len;
    if (next > end_)
      throw std::out_of_range("mmap readstring: exceeding memory-mapped area");

    addr_ = next;
    return std::string(str, len);
  }

  /// ReadString returns a string of the provided length from the current position and advances
  /// the position.
  std::string ReadString(size_t len)
  {
    if (addr_ + len > end_)
      throw std::out_of_range("mmap readstring: exceeding memory-mapped area");

    char* str = addr_;
    addr_ += len;
    return std::string(str, len);
  }


  // FIXME: drop MMapArray, use Address and Seek
#if 1
  /// Array returns a MMapArray of the specified primitive for a memory mapped reagion.
  template <typename _T, size_t _Rank>  MMapArray<_T, _Rank>
  Array(const size_t(&)[_Rank], const ssize_t(&)[_Rank], size_t offset = 0UL);

  template <typename _T, size_t _Rank>  MMapArray<_T, _Rank>
  Array(const std::array<size_t, _Rank>&, const std::array<ssize_t, _Rank>&, size_t offset = 0UL);


  /// Array returns a MMapArray of the specified primitive for a memory mapped reagion.
  template <typename _T, size_t _Rank>  MMapArray<_T, _Rank>
  Array(const size_t(&)[_Rank], size_t offset = 0UL);

  template <typename _T, size_t _Rank>  MMapArray<_T, _Rank>
  Array(const std::array<size_t, _Rank>&, size_t offset = 0UL);
#endif

  /// Align aligns the position to the next aligned position.
  void Align(int alignment)
  {
    uintptr_t p = reinterpret_cast<uintptr_t>(addr_);
    char* next = reinterpret_cast<char*>((p + alignment - 1) & ~(alignment - 1));

    if (next > end_)
      throw std::out_of_range("mmap align: exceeding memory-mapped area");

    addr_ = next;
  }

  /// Address returns the current position in the mmaped file -- FIXME is address of position?
  void* Address()                                        { return addr_; }

  /// Offset returns the offset of the mmaped region. --- FIXME??
  size_t Offset()
  {
    return reinterpret_cast<uintptr_t>(addr_) - reinterpret_cast<uintptr_t>(mmap_->Address());
  }

  /// Remaining returns the remaining size of the mmaped file from the current position.
  size_t Remaining()
  {
    return static_cast<size_t>(end_ - addr_);
  }

  /// Size returns the size of the view.
  size_t Size()                                           { return end_ - base_; }


  /// Seek advances the current position by len bytes.
  void Seek(ssize_t len)
  {
    if (addr_ + len > end_ || addr_ + len < base_)
      throw std::out_of_range("mmap seek: exceeding memory-mapped area");

    addr_ += len;
  }

 private:
  std::shared_ptr<MMap> mmap_;
  char* base_;
  char* addr_;
  char* end_;
};

template <typename _T, size_t _Rank>
struct MMapArray
{
  /// Constructor for an array of a memory-mapped file for the given dimentions and strides.
  MMapArray(std::shared_ptr<MMap> mmap,
            const size_t(&dims)[_Rank],
            const ssize_t(&strides)[_Rank],
            size_t offset = 0UL)
    : mmap_(std::move(mmap)),
      offset_(offset),
      dims_(get_array<size_t, _Rank>(dims)),
      strides_(get_array<ssize_t, _Rank>(strides))
  {
    if (offset_ + Size() > mmap_->Size())
      throw std::out_of_range("array exceeding memory-mapped range");
  }

  /// Constructor for an array of a memory-mapped file for the given dimentions and strides.
  MMapArray(std::shared_ptr<MMap> mmap,
            const size_t(&dims)[_Rank],
            size_t offset = 0UL)
    : mmap_(std::move(mmap)),
      offset_(offset),
      dims_(get_array<size_t, _Rank>(dims)),
      strides_(make_strides<_T>(dims_))
  {
    if (offset_ + Size() > mmap_->Size())
      throw std::out_of_range("array exceeding memory-mapped range");
  }

  /// Constructor for an array of a memory-mapped file for the given dimentions (no padding).
  MMapArray(std::shared_ptr<MMap> mmap,
            const std::array<size_t, _Rank>& dims,
            size_t offset = 0UL)
    : mmap_(std::move(mmap)),
      offset_(offset),
      dims_(dims),
      strides_(make_strides<_T>(dims_))
  {
    if (offset_ + Size() > mmap_->Size())
      throw std::out_of_range("array exceeding memory-mapped range");
  }


  /// Constructor for an array of a memory-mapped file for the given dimentions (no padding).
  MMapArray(std::shared_ptr<MMap> mmap,
            const std::array<size_t, _Rank>& dims,
            const std::array<ssize_t, _Rank>& strides,
            size_t offset = 0UL)
    : mmap_(std::move(mmap)),
      offset_(offset),
      dims_(dims),
      strides_(strides)
  {
    if (offset_ + Size() > mmap_->Size())
      throw std::out_of_range("array exceeding memory-mapped range");
  }

  /// Size returns the size of the array in bytes.
  size_t Size()                                           { return strides_[0] * dims_[0]; }


  std::shared_ptr<MMap>       mmap_;
  size_t                      offset_;
  std::array<size_t, _Rank>   dims_;
  std::array<ssize_t, _Rank>  strides_;
};

#if 1
template <typename _T, size_t _Rank>  MMapArray<_T, _Rank>
inline MMapView::Array(const size_t(&dims)[_Rank],
                       const ssize_t(&strides)[_Rank],
                       size_t offset)
{
  auto arr = MMapArray<_T, _Rank>(mmap_, addr_ - static_cast<char*>(mmap_->Address()), dims, strides);
  addr_ += arr.Size();
  return arr;
}


template <typename _T, size_t _Rank>  MMapArray<_T, _Rank>
inline MMapView::Array(const std::array<size_t, _Rank>& dims,
                       const std::array<ssize_t, _Rank>& strides,
                       size_t offset)
{
  auto arr = MMapArray<_T, _Rank>(mmap_, dims, strides, addr_ - static_cast<char*>(mmap_->Address()));
  addr_ += arr.Size();
  return arr;
}

template <typename _T, size_t _Rank>  MMapArray<_T, _Rank>
inline MMapView::Array(const size_t(&dims)[_Rank], size_t offset)
{
  auto arr = MMapArray<_T, _Rank>(mmap_, dims, addr_ - static_cast<char*>(mmap_->Address()));
  addr_ += arr.Size();
  return arr;
}
#endif

template< class T1, class T2 >
constexpr bool operator==(const MemoryMapped<T1>& lhs, const MemoryMapped<T2>& rhs) noexcept;


} // namespace grid

#endif // GRID_TENSOR_MMAP_H
