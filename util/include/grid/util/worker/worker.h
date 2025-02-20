//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_UTIL_WORKER_WORKER_H
#define GRID_UTIL_WORKER_WORKER_H

#include <grid/util/ioevent.h>

namespace grid {

// WorkerJob Describe a single job managed by the Worker
// Note that all fields are initialize to 0 when constructed
class Job;
class Worker;
struct WorkerJob
{
  static const int kMaxFunctorSize = 200;

  // MUST BE FIRST ENTRY!
  IOEvent             ioevent_;
  std::atomic<bool>   ioevent_scheduled_;

  enum Reschedule
  {
    kKill = -1,
    kOnce = 0,      // single run
    kAgain = 1,     // reschedule
  };


  // dual-linked list
  WorkerJob*              next_;
  WorkerJob*              prev_;

  // the lock protects this section
  WorkerJob*              block_;
  WorkerJob*              yield_;
  Worker*                 worker_;
  WorkerQueue*            queue_;
  bool                    is_queued_;
  // end of locked section

  int                     max_duration_;
  TimePoint               scheduled_time_;

  uint64_t                context_;

  bool                    result_;
  bool                    running_;

  std::atomic<Reschedule> reschedule_;
  std::atomic<bool>       woken_;

  std::atomic<int>        refcount_;
  char                    function_buffer_[kMaxFunctorSize];
};

//
// Implementation
//

inline void Worker::SetCurrentJob(Job::Id id)
{
  CurrentJob::tls_current_job_id_ = id;
}


template <typename R, typename... Args>
inline Job::Id Worker::AllocateJob(R(&&function)(Args...), Args... args)
{
  auto f = std::bind(&function, std::forward<Args>(args)...);

  Job::Id id = AllocateJob(sizeof(f));
  if (id != Job::kInvalid)
    *reinterpret_cast<std::function<bool()>*>(GetFunctionPointer(id)) = f;
  return id;
}


template <typename R, typename C, typename... Args>
inline Job::Id Worker::AllocateJob(R(C::*function)(Args...), C& c, Args&&... args)
{
  auto f = std::bind(std::mem_fn(function), &c, args...);

  Job::Id id = AllocateJob(sizeof(f));
  if (id != Job::kInvalid)
    *reinterpret_cast<std::function<bool()>*>(GetFunctionPointer(id)) = f;
  return id;
}


template <typename F, typename... Args>
inline Job Worker::Post(F&& function, Args&&... args)
{
  Job::Id id = AllocateJob(std::forward<F>(function), std::forward<Args>(args)...);
  if (id != Job::kInvalid && !PostJob(id))
  {
    ReleaseJob(id);
    id = Job::kInvalid;
  }
  return Job(id);
}


template <typename F>
inline Job Worker::PostRunAfter(F&& function, const Job& job)
{
  Job::Id id = AllocateJob(std::forward<F>(function));
  if (id != Job::kInvalid && !PostJobRunAfter(id, job.GetId()))
  {
      ReleaseJob(id);
      id = Job::kInvalid;
  }
  return Job(id);
}


template <typename F>
inline Job Worker::PostNext(F&& function)
{
  Job::Id id = AllocateJob(std::forward<F>(function));
  if (id != Job::kInvalid && !PostJobNext(id))
  {
    ReleaseJob(id);
    id = Job::kInvalid;
  }
  return Job(id);
}


template <typename F>
inline Job Worker::PostImmediate(F&& function)
{
  Job::Id id = AllocateJob(std::forward<F>(function));
  if (id != Job::kInvalid && !PostJobImmediate(id))
  {
    ReleaseJob(id);
    id = Job::kInvalid;
  }
  return Job(id);
}


template <typename F>
inline Job Worker::PostAtTime(F&& function, TimePoint time)
{

  Job::Id id = AllocateJob(std::forward<F>(function));
  if (id != Job::kInvalid && !PostJobAtTime(id, time))
  {
    ReleaseJob(id);
    id = Job::kInvalid;
  }
  return Job(id);
}


template <typename F>
inline Job Worker::PostDelayedMsec(F&& function, int msec)
{
  Job::Id id = AllocateJob(std::forward<F>(function));
  if (id != Job::kInvalid && !PostJobDelayedMsec(id, msec))
  {
    ReleaseJob(id);
    id = Job::kInvalid;
  }
  return Job(id);
}


template <typename F>
inline Job Worker::PostSleeping(F&& function)
{
  return Worker::PostAtTime(std::forward<F>(function), kInfiniteTime);
}


} // end of namespace grid

#endif  // GRID_UTIL_WORKER_WORKER_H
