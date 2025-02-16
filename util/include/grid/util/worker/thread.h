//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_UTIL_WORKER_THREAD_H
#define GRID_UTIL_WORKER_THREAD_H

#include <grid/util/thread.h>

namespace grid {

/// WorkerThread describes a single thread in a WorkerQueue.
class WorkerThread
{
  friend class Worker;

 public:
  WorkerThread(const std::string& name, Worker* worker, WorkerQueue* queue);
  WorkerThread(const WorkerThread& other) = delete;
  WorkerThread(WorkerThread&& other);
  ~WorkerThread();

  /// Start the thread
  void Start();

  /// Kill the thread and wait for completion(!)
  void Kill();

  /// Set thread to 'running', but still needs to be awaken by the queue cond.
  void SetRunning();

  /// Set thread asleep
  void SetSleeping();


 private:
  enum State
  {
    kStateSleeping = 0,
    kStateRunning,
    kStateKilled,
  };

  const std::string   name_;
  Worker*             worker_;
  WorkerQueue*        queue_;

  Thread*             thread_;
  std::atomic<State>  state_;
};

//
// Implementations
//

inline void WorkerThread::Kill()
{
  state_ = kStateKilled;
  queue_->wait_cond_.notify_all();

  // TODO: kill the thread to cancel a blocking thread?
}

inline void WorkerThread::SetRunning()
{
  auto expected = kStateSleeping;
  state_.compare_exchange_strong(expected, kStateRunning);
}


inline void WorkerThread::SetSleeping()
{
  state_ = kStateSleeping;
}

} // end of namespace grid

#endif  // GRID_UTIL_WORKER_THREAD_H
