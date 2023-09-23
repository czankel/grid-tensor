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
  MMap() = default;

  /// Constructor for memory-mapped file specified by the file name/path.
  MMap(const std::string& name);

  /// Constructor for memory-mapped file specified by file-descriptor and memory-mapped size.
  MMap(int fd, size_t file_size)
    : fd_(fd),
      addr_(static_cast<char*>(mmap(NULL, file_size, PROT_READ, MAP_FILE | MAP_SHARED, fd_, 0))),
      file_size_(file_size)
  {}

  /// Copy constructor.
  MMap(const MMap& other)
    : fd_(dup(other.fd_)),
      addr_(static_cast<char*>(mmap(NULL, other.Size(), PROT_READ, MAP_FILE | MAP_SHARED, fd_, 0))),
      file_size_(other.Size())
  {
    if (addr_ == MAP_FAILED)
      throw("mmap failed");
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
    if (addr_ != nullptr && file_size_ > 0)
      ::munmap(addr_, file_size_);
    if (fd_ != 0)
      ::close(fd_);
  }


  // Size returns the size of the mmaped file
  size_t Size() const                                     { return file_size_; }

  // Address returns the address of the mmaped file
  void* Address() const                                   { return addr_; }

  // End of the mmaped region
  void* End() const                                       { return addr_ + file_size_; }


 protected:
  int     fd_;
  char*   addr_;
  size_t  file_size_;
};


/// MMapView provides a "view" into a memory-mmaped file and includes a current position for
/// sequential "read" operations.
class MMapView
{
 public:
  MMapView(std::shared_ptr<MMap> mmap)
    : mmap_(mmap),
      addr_(static_cast<char*>(mmap->Address())),
      pos_(static_cast<char*>(mmap->Address())),
      end_(static_cast<char*>(mmap->End()))
  {}

  /// Read returns the value of the specified type at the current position and advances the position.
  /// The current position may be unaligned.
  template<typename T>
  T Read()
  {
    char* next = pos_ + sizeof(T);
    if (next > end_)
      throw std::out_of_range("mmap read: exceeding memory-mapped area");

    T temp;
    memcpy(&temp, pos_, sizeof(T));
    pos_ = next;
    return temp;
  }

  /// Read copies the data to the provided value from the current position and advances the position.
  /// The current position may be unaligned.
  template<typename T>
  void Read(T& temp)
  {
    char* next = pos_ + sizeof(T);
    if (next > end_)
      throw std::out_of_range("mmap read: exceeding memory-mapped area");

    memcpy(&temp, pos_, sizeof(T));
    pos_ = next;
  }

  /// Read copies data from the current position to the provided destination with the provided lenght.
  template<typename T>
  void Read(T* dest, size_t len)
  {
    char* next = pos_ + len;
    if (next > end_)
      throw std::out_of_range("mmap read: exceeding memory-mapped area");

    memcpy(reinterpret_cast<char*>(dest), pos_, len);
    pos_ = next;
  }

  /// ReadString returns a std::string at the current position encoded as lenght, string and
  /// advances the position.
  std::string ReadString()
  {
    char* next = pos_ + 1;
    if (next >  end_)
      throw std::out_of_range("mmap readstring: exceeding memory-mapped area");

    uint32_t len;
    memcpy(&len, pos_, sizeof(uint32_t));

    char* str = next;
    next += len;
    if (next > end_)
      throw std::out_of_range("mmap readstring: exceeding memory-mapped area");

    pos_ = next;
    return std::string(str, len);
  }

  /// ReadString returns a string of the provided length from the current position and advances
  /// the position.
  std::string ReadString(size_t len)
  {
    if (pos_ + len > end_)
      throw std::out_of_range("mmap readstring: exceeding memory-mapped area");

    char* str = pos_;
    pos_ += len;
    return std::string(str, len);
  }

  /// Array returns a MMapArray of the specified primitive for a memory mapped reagion.
  template <typename _T, size_t _Rank>  MMapArray<_T, _Rank>
  Array(size_t offset, const size_t(&)[_Rank], const ssize_t(&)[_Rank]);

  template <typename _T, size_t _Rank>  MMapArray<_T, _Rank>
  Array(size_t offset, const std::array<size_t, _Rank>&, const std::array<ssize_t, _Rank>&);

  /// Align aligns the position to the next aligned position.
  void Align(int alignment)
  {
    uintptr_t p = reinterpret_cast<uintptr_t>(pos_);
    char* next = reinterpret_cast<char*>((p + alignment - 1) & ~(alignment - 1));

    if (next > end_)
      throw std::out_of_range("mmap align: exceeding memory-mapped area");

    pos_ = next;
  }
  
  // Address returns the base address of the view
  void* Address() const                                   { return addr_; }

  /// Position returns the current position in the mmaped file
  void* Position()                                        { return pos_; }

  /// Offset returns the offset of the mmaped region.
  size_t Offset()
  {
    return reinterpret_cast<uintptr_t>(pos_) - reinterpret_cast<uintptr_t>(mmap_->Address());
  }

  /// Remaining returns the remaining size of the mmaped file from the current position.
  size_t Remaining()
  {
    return static_cast<size_t>(end_ - pos_);
  }

  /// Size returns the size of the view.
  size_t Size()                                           { return end_ - addr_; }


  /// Seek advances the current position by len bytes.
  void Seek(ssize_t len)
  {
    if (pos_ + len > end_ || pos_ + len < addr_)
      throw std::out_of_range("mmap seek: exceeding memory-mapped area");

    pos_ += len;
  }

 private:
  std::shared_ptr<MMap> mmap_;
  char* addr_;
  char* pos_;
  char* end_;
};

template <typename _T, size_t _Rank>
struct MMapArray
{
  /// Constructor for an array of a memory-mapped file for the given dimentions and strides.
  MMapArray(std::shared_ptr<MMap> mmap,
            size_t offset,
            const size_t(&dims)[_Rank],
            const ssize_t(&strides)[_Rank])
    : mmap_(std::move(mmap)),
      offset_(offset),
      dims_(get_array<size_t, _Rank>(dims)),
      strides_(get_array<ssize_t, _Rank>(strides))
  {
    if (offset_ + Size() > mmap_->Size())
      throw std::out_of_range("array exceesing memory-mapped range");
  }

  /// Constructor for an array of a memory-mapped file for the given dimentions (no padding).
  MMapArray(std::shared_ptr<MMap> mmap,
            size_t offset,
            const std::array<size_t, _Rank>& dims)
    : mmap_(std::move(mmap)),
      offset_(offset),
      dims_(dims),
      strides_(get_strides<_T>(dims_))
  {
    if (offset_ + Size() > mmap_->Size())
      throw std::out_of_range("array exceesing memory-mapped range");
  }


  /// Constructor for an array of a memory-mapped file for the given dimentions (no padding).
  MMapArray(std::shared_ptr<MMap> mmap, size_t offset,
            const std::array<size_t, _Rank>& dims,
            const std::array<ssize_t, _Rank>& strides)
    : mmap_(std::move(mmap)),
      offset_(offset),
      dims_(dims),
      strides_(strides)
  {
    if (offset_ + Size() > mmap_->Size())
      throw std::out_of_range("array exceesing memory-mapped range");
  }

  /// Size returns the size of the array in bytes.
  size_t Size()                                           { return strides_[0] * dims_[0]; }


  std::shared_ptr<MMap>       mmap_;
  size_t                      offset_;
  std::array<size_t, _Rank>   dims_;
  std::array<ssize_t, _Rank>  strides_;
};


template <typename _T, size_t _Rank>  MMapArray<_T, _Rank>
inline MMapView::Array(size_t offset,
                       const size_t(&dims)[_Rank],
                       const ssize_t(&strides)[_Rank])
{
  auto arr = MMapArray<_T, _Rank>(mmap_, pos_ - static_cast<char*>(mmap_->Address()), dims, strides);
  pos_ += arr.Size();
  return arr;
}


template <typename _T, size_t _Rank>  MMapArray<_T, _Rank>
inline MMapView::Array(size_t offset,
                       const std::array<size_t, _Rank>& dims,
                       const std::array<ssize_t, _Rank>& strides)
{
  auto arr = MMapArray<_T, _Rank>(mmap_, pos_ - static_cast<char*>(mmap_->Address()), dims, strides);
  pos_ += arr.Size();
  return arr;
}


} // namespace grid

#endif // GRID_TENSOR_MMAP_H

