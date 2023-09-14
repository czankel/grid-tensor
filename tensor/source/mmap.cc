//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/mmap.h>

namespace grid {

MMap::MMap(const std::string& name)
{
  fd_ = open(name.c_str(), O_RDONLY);
  if (fd_ < 0)
    throw("no such file: " + name);

  file_size_ = lseek(fd_, 0, SEEK_END);
  void* addr = mmap(NULL, file_size_, PROT_READ, MAP_FILE | MAP_SHARED, fd_, 0);
  if (addr == MAP_FAILED)
    throw("mmap failed");

  addr_ = reinterpret_cast<char*>(addr);
}

} // end of namespace grid
