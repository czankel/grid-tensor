//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_UTIL_WORKER_QUEUE_H
#define GRID_UTIL_WORKER_QUEUE_H

namespace grid {

// WorkerQueue describes a Job queue managed by at least one thread.
//
// The current implementation uses a single thread per queue. FIXME??
struct WorkerJob;
struct WorkerQueue
{
  WorkerQueue() :
      ready_(NULL),
      scheduled_(&ready_),
      sleeping_(&ready_),
      needs_reschedule_(false)
  {}

  WorkerQueue(const WorkerQueue& other) :
      ready_(other.ready_),
      scheduled_(&ready_),
      sleeping_(&ready_),
      needs_reschedule_(false)
  {}

  std::mutex          wait_mutex_;
  std::condition_variable  wait_cond_;

  mutable std::mutex  mutex_;     // lock for accessing the job list
  WorkerJob*          ready_;     // ready to run
  WorkerJob**         scheduled_; // scheduled at a certain time
  WorkerJob**         sleeping_;

  std::atomic<bool>        needs_reschedule_;
};

} // end of namespace grid

#endif  // GRID_UTIL_WORKER_QUEUE_H
