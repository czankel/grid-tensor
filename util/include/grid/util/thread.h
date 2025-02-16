//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_UTIL_THREAD_H
#define GRID_UTIL_THREAD_H

#include <pthread.h>
#include <string>
#include <functional>

#ifdef ANDROID
#include <jni.h>
#endif

namespace grid {


class Thread   // noncopiable??
{
 public:
  ~Thread();

  /// Thread creates a new thread for the provided function. The
  template <typename F>
  Thread(const F& function) :
    name_("anonymous"),
    function_(function),
    handle_(0)
  {
  }

  template <typename F>
  Thread(const std::string& name, const F& function) :
    name_(name),
    function_(function),
    handle_(0)
  {
  }

  // Start thread
  bool Start();

  // Join thread
  void* Join();

#ifdef ANDROID
  /// Set the JavaVM.
  /// This function must be called once, and before any thread is created.
  static void SetJavaVM(JavaVM* java_vm);

  // Destroy the current thread. This function. This is an internal function.
  void DestroyThread();
#endif

 private:
  static void* Run(void* data);

  const std::string     name_;
  std::function<void*()>     function_;
  pthread_t             handle_;

#ifdef ANDROID
  JNIEnv*               jni_env_;
  static JavaVM*        java_vm_;
  static pthread_key_t  thread_key_;
#endif
};

class CurrentThread
{
 public:
  static inline void SleepMsec(unsigned int msec)
  {
    const timespec ts =
    {
      (time_t) msec / 1000,
      ((time_t) msec % 1000) * 1000000L,
    };
    nanosleep(&ts, NULL);
  }
};

} // end of namespace grid

#endif  // GRID_UTIL_THREAD_H
