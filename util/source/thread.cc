//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/util/thread.h>

namespace grid {

#ifdef ANDROID
// global JavaVM handle
JavaVM* Thread::java_vm_;

pthread_key_t	Thread::thread_key_;

static void _DestroyThread(void* value)
{
  Thread* thread = reinterpret_cast<Thread*>(value);
  if (thread != 0)
    thread->DestroyThread();
}

void Thread::SetJavaVM(JavaVM* java_vm)
{
  java_vm_ = java_vm;
}

void Thread::DestroyThread()
{
  if (java_vm_ != 0 && jni_env_ != 0)
  {
    java_vm_->DetachCurrentThread();
    pthread_setspecific(thread_key_, NULL);
    jni_env_ = NULL;
  }
}
#endif

void* Thread::Run(void* data)
{
  Thread* thread = static_cast<Thread*>(data);

  if (thread == 0)
    return 0;

#ifdef ANDROID
  // note that we ensure java_vm_ is not null in Start
  if (thread->java_vm_->AttachCurrentThread(&thread->jni_env_, NULL) != 0)
    return 0;
  pthread_key_create(&thread_key_, _DestroyThread);
#endif

  return thread->function_();
}

bool Thread::Start()
{
  return
#ifdef ANDROID
      java_vm_ != 0 &&
#endif
      pthread_create(&handle_, 0, &Thread::Run, (void*)this) == 0;
}

void* Thread::Join()
{
  void* retval = NULL;

  if (handle_ != 0)
  {
    pthread_join(handle_, &retval);
    handle_ = 0;
  }
  return retval;
}

grid::Thread::~Thread()
{
  if (handle_ != 0)
    pthread_detach(handle_);
  // FIXME: disable join, mark as joined, use mutexes to do this
}

}
