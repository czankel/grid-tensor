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

#include "array.h"

namespace grid {

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

  /// Array returns an ArrayView of the specified primitive for a memory mapped reagion.
  template <typename T, size_t TRank>
  ArrayView<T, TRank> Array(const size_t(&dimensions)[TRank], const ssize_t(&strides)[TRank])
  {
    auto arr = ArrayView<T, TRank>(reinterpret_cast<T>(addr_), dimensions, strides);
    addr_ += arr.Size();
    return arr;
  }

  template <typename T, size_t TRank>
  ArrayView<T, TRank>
  Array(const std::array<size_t, TRank>& dimensions, const std::array<ssize_t, TRank>& strides)
  {
    auto arr = ArrayView<T, TRank>(reinterpret_cast<T*>(addr_), dimensions, strides);
    addr_ += arr.Size();
    return arr;
  }

  /// Array returns an ArrayView of the specified primitive for a memory mapped reagion.
  template <typename T, size_t TRank>
  ArrayView<T, TRank>
  Array(const size_t(&dimensions)[TRank])
  {
    auto arr = ArrayView<T, TRank>(reinterpret_cast<T*>(addr_), dimensions);
    addr_ += arr.Size();
    return arr;
  }

  template <typename T, size_t TRank>
  ArrayView<T, TRank>
  Array(const std::array<size_t, TRank>& dimensions)
  {
    auto arr = ArrayView<T, TRank>(reinterpret_cast<T>(addr_), dimensions);
    addr_ += arr.Size();
    return arr;
  }


  /// Align aligns the position to the next aligned position.
  void Align(int alignment)
  {
    uintptr_t p = reinterpret_cast<uintptr_t>(addr_);
    char* next = reinterpret_cast<char*>((p + alignment - 1) & ~(alignment - 1));

    if (next > end_)
      throw std::out_of_range("mmap align: exceeding memory-mapped area");

    addr_ = next;
  }

  /// Address returns the current position in the mmaped file
  void* Address()                                        { return addr_; }

  /// Offset returns the offset of the mmaped region.
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
  size_t Size() const                                     { return end_ - base_; }


  /// Seek advances the current position by len bytes and returns the current position.
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

} // namespace grid

#endif // GRID_TENSOR_MMAP_H
