//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_UTIL_WORKER_JOB_H
#define GRID_UTIL_WORKER_JOB_H

#include <grid/util/ioevent.h>

#include "worker.h"

namespace grid {


inline Worker& Job::GetWorker()
{
  if (!IsValid())
    throw std::runtime_error("invalid job");
  return *reinterpret_cast<WorkerJob*>(id_)->worker_;
}

inline Worker& Job::GetWorker() const
{
  if (!IsValid())
    throw std::runtime_error("invalid job");
  return *reinterpret_cast<WorkerJob*>(id_)->worker_;
}


//
// CurrentJob::
//

inline bool CurrentJob::IsValid()
{
  return tls_current_job_id_ != Job::kInvalid;
}

inline Worker& CurrentJob::GetWorker()
{
  if (!IsValid())
    throw std::runtime_error("invalid job");
  return *reinterpret_cast<WorkerJob*>(tls_current_job_id_)->worker_;
}

inline Job CurrentJob::GetJob()
{
  return Job(tls_current_job_id_);
}


inline void CurrentJob::SetContext(uint64_t context)
{
  if (IsValid())
    GetWorker().SetContext(tls_current_job_id_, context);
}


inline bool CurrentJob::Reschedule()
{
  return IsValid() &&
    GetWorker().Reschedule(tls_current_job_id_);
}


inline bool CurrentJob::NeedsReschedule()
{
  return IsValid() &&
    GetWorker().NeedsReschedule(tls_current_job_id_);
}


inline bool CurrentJob::RescheduleAtTime(TimePoint time)
{
  return IsValid() &&
    GetWorker().RescheduleAtTime(tls_current_job_id_, time);
}


inline bool CurrentJob::RescheduleDelayedMsec(int msec)
{
  return IsValid() &&
    GetWorker().RescheduleDelayedMsec(tls_current_job_id_, msec);
}


inline bool CurrentJob::RescheduleAfterJob(const Job& job, bool inherit_priority)
{
  return IsValid() &&
    GetWorker().RescheduleAfterJob(tls_current_job_id_, job.id_, inherit_priority);
}


inline bool
CurrentJob::RescheduleAfterEvent(IOEvent::Type type,const IOEvent& ioevent)
{
  return IsValid() &&
    GetWorker().RescheduleAfterEvent(tls_current_job_id_, type, ioevent);
}


inline void CurrentJob::Kill()
{
  if (IsValid())
    GetWorker().KillJob(Job(tls_current_job_id_));
}


inline bool CurrentJob::IsRescheduled()
{
  return IsValid() &&
    GetWorker().IsRescheduled(Job(tls_current_job_id_));
}


inline bool CurrentJob::IsRescheduledWaiting()
{
  return IsValid() &&
    GetWorker().IsRescheduledWaiting(Job(tls_current_job_id_));
}


inline bool CurrentJob::RescheduleSleeping()
{
  return IsValid() &&
    GetWorker().RescheduleAtTime(tls_current_job_id_, kInfiniteTime);
}


inline void CurrentJob::WakeBlocked()
{
  if (IsValid())
    GetWorker().WakeBlocked(Job(tls_current_job_id_));
}

//
// Implementations
//

inline Job::Job(const Job& other) : id_(other.id_)
{
  if (id_ != kInvalid)
    GetWorker().AddRefJob(id_);
}


inline Job::~Job()
{
  if (id_ != kInvalid)
  {
    GetWorker().ReleaseJob(id_);
    id_ = kInvalid;
  }
}


inline Job& Job::operator=(const Job& other)
{
  if (IsValid())
    GetWorker().ReleaseJob(id_);

  id_ = other.id_;

  if (other.IsValid())
    GetWorker().AddRefJob(id_);

  return *this;
}


inline void Job::Release()
{
  if (id_ != kInvalid)
    GetWorker().ReleaseJob(id_);
  id_ = kInvalid;
}

inline bool Job::Wake()
{
  return GetWorker().Wake(*this); // FIXME IsValid?
}

inline void Job::Kill()
{
  GetWorker().KillJob(*this);
}

inline void Job::ChangeScheduledTime(TimePoint time)
{
  GetWorker().RescheduleAtTime(id_, time);
}

inline void Job::SetContext(uint64_t context)
{
  GetWorker().SetContext(id_, context);
}

inline Job::Status Job::GetStatus() const
{
  return GetWorker().GetJobStatus(id_);
}

inline Job::Job(const Id id) : id_(id)
{
  if (id_ != kInvalid)
    GetWorker().AddRefJob(id_);
}

} // end of namespace grid

#endif  // GRID_UTIL_WORKER_JOB_H
