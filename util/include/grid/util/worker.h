//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_UTIL_WORKER_H
#define GRID_UTIL_WORKER_H

#include <stdint.h>
#include <time.h>

#include <functional>
#include <chrono>
#include <list>
#include <mutex>
#include <new>

#include <grid/util/ioevent.h>
#include <grid/util/thread.h>

namespace grid {

using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;
using SteadyClock = std::chrono::steady_clock;
static const TimePoint kInfiniteTime = TimePoint::max();

class Worker;
class WorkerThread;
struct WorkerJob;
struct WorkerQueue;

/// Job is a scheduling entity for a worker.
///
/// The job describes a function that is executed when it is scheduled
/// by the worker. The function are not interrupted and are rescheduled
/// after they return.
///
/// The job includes many function that allow to reschedule with a variety
/// of options (delayed, at-time, waiting for another job)

class Job
{
 public:

  /// Type of the job dentifier type.
  using Id = uintptr_t;

  /// Status represents the state of a job.
  enum Status
  {
    kInvalid = 0,   //< Job is invalid
    kRunning,       //< Job is running
    kWaiting,       //< Job is waiting
    kDone,          //< Job has completed successfully
    kError          //< Job has encountered an error
  };

  inline Job() : id_(kInvalid) {}


  /// Copy constructor.
  ///
  /// @param[in] job  Other job.
  Job(const Job& other);

  ~Job();

  /// Assign operator.
  ///
  /// @param[in] other  Other job.
  /// @returns This job.
  Job& operator=(const Job& other);

  /// not-operator to check if job is valid
  bool operator!()                                { return id_ == kInvalid; }

  /// IsValid returns if the given job is valid.
  ///
  /// @returns True if job is valid or false, otherwise.
  bool IsValid() const                            { return id_ != kInvalid; }

  /// GetStatus returns the current status of the job.
  ///
  /// Note that there's a race condition, by the time the caller processes
  /// the information, the status might have changed.
  ///
  /// @returns Current status of the job.
  Status GetStatus() const;

  /// GetId returns the job identifier.
  ///
  /// @returns Job identifier
  inline Id GetId() const                         { return id_; }

  /// GetWorker returns the Worker for the job.
  ///
  /// Worker is only valid while the Job is valid.
  ///
  /// @returns Worker
  /// @throws runtime_error if invalid job
  Worker& GetWorker();
  Worker& GetWorker() const;

  /// GetContext returns the job's context id.
  ///
  /// The context id is an unsigned long value that is inherited when
  /// creating a new job.
  ///
  /// @returns Context id.
  uint64_t GetContext() const;

  /// SetContext tests the job's context id.
  ///
  /// Note that this will not change the current context id if the job is
  /// running. Use 'CurrentJob::SetContext()' for this.
  ///
  /// @param[in] context  Context for the job
  void SetContext(uint64_t context);


  /// Wake wakes up the job.
  ///
  /// This function is a no-op if the job is already running
  ///
  /// @returns false if invalid or killed.
  bool Wake();

  /// Kill kills the job.
  ///
  /// The job will be removed if it is queued and currently not running,
  /// or won't be scheduled once the currently running function returns.
  ///
  /// Note that the job cannot be rescheduled once it is 'killed'.
  void Kill();

  /// ChangeScheduledTime changes the reschedule time for the job.
  ///
  /// @param[in] time  New scheduled time for the job.
  void ChangeScheduledTime(TimePoint time);

  /// Release releases the job and marks it 'invalid'. The Job must not be
  /// used anymore after calling this function.
  void Release();

 private:
  friend class Worker;
  friend class CurrentJob;

  Job(const Id id);

  Id      id_;
};

/// The CurrentJob describes the current job that has been set for the current
/// thread. If the thread is not scheduled by the worker, the current job
/// is invalid.
///
/// To get the current job, use:
///
///   Job job = CurrentJob::GetJob();

class CurrentJob
{
  friend class Worker;

  CurrentJob();

 public:

  /// GetWorker is a staic function to return the current worker of the thread
  static Worker& GetWorker();

  /// GetJob is a static function to return the current job for the thread
  /// of the caller.
  ///
  /// @returns Current job or kInvalid if none was set.
  static Job GetJob();

  /// IsValid returns if the current thread is a valid job
  static bool IsValid();

  /// SetContext sets the context id of the current job.
  ///
  /// @param[in] context  New context for the job.
  static void SetContext(uint64_t context);

  /// GetCOntext returns the context id of the current job.
  ///
  /// @returns  Context of the current job.
  static uint64_t GetContext();


  /// Reschedule reschedules the current job to run after all scheduled job.
  ///
  /// @returns True for success or false if job is invalid or has been killed.
  static bool Reschedule();

  /// RescheduleDelayedMsec reschedules the current job to run after the
  /// specified delay from now.
  ///
  /// Setting the delay to kInifiniteTime set the job to sleep.
  ///
  /// @params[in] mdelay  Delay in miliseconds or Worker::kInfiniteTime
  /// @returns True for success or false if job is invalid or has been killed.
  static bool RescheduleDelayedMsec(int msec);

  /// RescheduleAtTime reschedules the current job to run at a later  time.
  ///
  /// @params[in] time  Time when this job should run again.
  /// @returns True for success or false if job is invalid or has been killed.
  static bool RescheduleAtTime(TimePoint time);

  /// RescheduleAfterJob reschedules the current job to run after the specified
  /// job has completed.
  ///
  /// The specified job won't change its priority (and might continue to sleep),
  /// unless inherit_priority is set, in which case the specified job will
  /// inherit the priority of the current job.
  ///
  /// There is currently no mechanism to wait for multiple jobs.
  ///
  /// @param[in] function  Function to execute
  /// @returns True for success or false if job is invalid or has been killed.
  static bool RescheduleAfterJob(const Job& job, bool inherit_priority = false);

  /// RescheduleSleeping sets job to sleep.
  ///
  /// (This is the same as RescheduleDelayedMsec(Worker::kInfiniteTime))
  /// @returns True for success or false if job is invalid or has been killed.
  static bool RescheduleSleeping();

  /// Set job asleep and wake it for the specified event.
  ///
  /// The job is reschedule when the specified event occurs or when
  /// the job is woken up.
  ///
  /// @param[in] type  IO Event type
  /// @param[in] ioevent  IO Event
  /// @returns True for success or false if job is invalid or has been killed.
  static bool RescheduleAfterEvent(IOEvent::Type type, const IOEvent& ioevent);


  /// IsRescheduled returns if the current job is still alive and hasn't been
  /// killed.
  ///
  /// @returns true if the current job is still alive.
  static bool IsRescheduled();

  /// IsRescheduledWaiting returns if the specified job is rescheduled
  /// for a later time or is blocked by another job.
  ///
  /// @returns true if the job is waiting.
  static bool IsRescheduledWaiting();

  /// NeedsReschedule checks if another higher priority job is waiting in the
  /// same queue of the current job.
  ///
  /// @param[in] id  Job id.
  /// @returns True if the worker would need to reschedule the queue.
  static bool NeedsReschedule();


  /// Kill kills the current job.
  ///
  /// It won't be scheduled again once the current function returns.
  ///
  /// Note that the job cannot be rescheduled once it is 'killed'.
  static void Kill();

  /// WakeBlocked wakes all blocked jobs.
  static void WakeBlocked();

 private:
  static inline Job GetCurrentJob()
  {
    return Job(tls_current_job_id_);
  }

  static __thread Job::Id tls_current_job_id_;
};

/// Worker is an interface for managing work queue of uninterruptible and
/// cooperative jobs.
///
/// Jobs consist of a function that is called when the job is ready to run.
/// They can be scheduled or rescheduled as follows:
///
///  - immediately or when any currently running job returns
///  - next after the current job gets rescheduled
///  - after all currently available jobs
///  - at a specific time
///  - after another job has completed.
///
/// Jobs can be in different queues. Each queue is managed by a queue of
/// threads.

class Worker
{
  friend class WorkerThread;

  static const int kMaxEvents = 1000;

 public:

  static const int kDefaultDuration = 0;  //< Default run-time duration
  static constexpr int kMaxJobCount = 100;
  static constexpr unsigned int kDefaultConcurrentThreadCount = 1;

  Worker (const Worker&) = delete;
  Worker(Worker&&) = delete;
  Worker& operator= (const Worker&) = delete;

  Worker(bool no_threads = false, bool event_thread = true, unsigned int job_max_count = kMaxJobCount);

  ~Worker();

  /// Run runs a no-thread worker until the last job exits.
  bool Run();

  /// Stop stops the worker.
  void Stop();


  /// SetMaxConcurrentThreadCount sets the maximum number of concurrent
  /// threads.
  ///
  /// @params[in] count  Max concurrently running threads
  /// @returns True if thread count was changed
  bool SetMaxConcurrentThreadCount(unsigned int count);

  /// GetMaxConcurrentThreadCount returns the maximum number of concurrent
  /// threads.
  ///
  /// @returns max concurrent thread count
  unsigned int GetMaxConcurrentThreadCount();


  /// Wake wakes the specified job.
  ///
  /// @param[in] job  Job to wake.
  /// @returns True for success or false if job is invalid or has been killed.
  bool Wake(const Job& job);

  /// WakeBlocked Wakes all jobst that are blocked on the specified job.
  ///
  /// @param[in] job Job that is  blocking other jobs.
  void WakeBlocked(const Job& job);


  /// KillJob kills the specified job.
  ///
  /// Note that the job might still be running when this function returns.
  /// Use WaitForJob() to wait for completion.
  /// Also note that the job cannot be rescheduled once it has been 'killed'.
  ///
  /// @param[in] job  Job to kill.
  void KillJob(const Job& job);


  /// IsRescheduled returns if the current job is still alive and hasn't been
  /// killed.
  ///
  /// @returns true if job is still alive.
  bool IsRescheduled(const Job& job);

  /// IsRescheduledWaiting returns if the specified job is rescheduled
  /// for a later time or is blocked by another job.
  ///
  /// @param[in] job  Job.
  /// @returns True if job has been rescheduled or false for errors.
  bool IsRescheduledWaiting(const Job& job);


  /// Wait for a job to complete
  /// Note that jobs that are rescheduled are not complete
  /// @param[in] job to wait for
  /// @returns true if job has completed or false if wait was canceled
  bool WaitForJob(const Job& job);

  /// Cancel any wait for a specific job to complete
  /// @param[in] job for someone to wait for
  void CancelWaitForJob(const Job& job);


  /// Post posts a new job to the current queue.
  ///
  /// @param[in] function  Function to post; see function.h
  /// @returns Job identification; use IsValid() to check if it was successful
  template <typename F, typename... Args> Job Post(F&&, Args&&...);

  /// PostImmediate posts a job to run immediately.
  ///
  /// @param[in] function  Function to post.
  /// @returns Job identification; use IsValid() to check if it was successful
  template <typename F> Job PostImmediate(F&& function);

  /// PostNext posts a job to run after the current job.
  ///
  /// @param[in] function  Function to post; see function.h
  /// @returns Job identification; use IsValid() to check if it was successful
  template <typename F> Job PostNext(F&& function);

  /// PostDelayedMsec posts a job to run after the specified delay.
  ///
  /// @param[in] function  Function to post; see function.h
  /// @param[in] mdealy  Delay in [ms] or kInifinteTime
  /// @returns Job identification; use IsValid() to check if it was successful
  template <typename F> Job PostDelayedMsec(F&& function, int msec);

  /// PostAtTime posta a job to run at the specific time.
  ///
  /// @param[in] function  Function to post; see function.h
  /// @param[in] time  Time when the job should run.
  /// @returns Job identification; use IsValid() to check if it was successful
  template <typename F> Job PostAtTime(F&& function, TimePoint time);

  /// PostRunAfter posts a job to run after the specified job has completed.
  ///
  /// Unlike the other Post methods, the added job has a dependency and won't
  /// run until the parent job has finished. The posted job is started
  /// in the state 'sleeping'
  ///
  /// @param[in] function  Function that should be run
  /// @param[in] job  Dependent job that must complete
  /// @returns Job identification; use IsValid() to check if it was successful
  template <typename F> Job PostRunAfter(F&& function, const Job& job);

  /// PostSleeping posts a job as 'sleeping', waiting for a wake-up event
  ///
  /// @param[in] function  Function that should be run.
  template <typename F> Job PostSleeping(F&& function);

 protected:

  friend class Job;
  friend class CurrentJob;

  /// SetCurrentJob sets the thread-local variable to the current job.
  ///
  /// @param[in] id  Job id for the current job.
  void SetCurrentJob(Job::Id id);


  /// AllocateJob allocates a new job.
  ///
  /// @param[in] size  Size required for the job function.
  /// @returns New job id or kInvalid for errors.
  Job::Id AllocateJob(size_t size);


  /// Return the 'function' pointer for the specific id.
  ///
  /// The function pointer is used to 're-initialize' a job and it may point
  /// to a pre-allocated array.
  ///
  /// @param[in] id  Job id.
  /// @returns Pointer to the job function or nullptr for errors.
  void* GetFunctionPointer(const Job::Id id);

  /// AddRefJob increments the reference counter for the job.
  ///
  /// @param[in] id  Job id.
  void AddRefJob(Job::Id id);

  /// ReleaseJob releases the job and decrements the reference counter. The
  /// job is deleted when the count goes to 0.
  ///
  /// @param[in] id  Job id.
  void ReleaseJob(Job::Id id);


  /// SetContext sets the context id for the provided job.
  ///
  /// @param[in] id  Job id.
  /// @param[in] context  Context.
  void SetContext(Job::Id id, uint64_t context);

  /// GetContext returns the context id for the provided job.
  ///
  /// @param[in] id  Job id.
  /// @returns Context of the job.
  uint64_t GetContext(Job::Id id) const;


  /// GetJobStatus returns the status of the specified job.
  ///
  /// @param[in] id  Job id.
  /// @returns Status of the job.
  Job::Status GetJobStatus(Job::Id id) const;


  /// NeedsReschedule returns if the worker would need to reschedule the queue.
  ///
  /// NeedsReschedule checks if another higher priority job is waiting in the
  /// same queue of the specified job.
  ///
  /// @param[in] id  Job id.
  /// @returns True if the worker would need to reschedule the queue.
  bool NeedsReschedule(Job::Id id) const;


  /// Post posts a job to be added to the list of schedulable jobs.
  ///
  /// @param[in] id  Job id.
  /// @returns True if scheduled successfully.
  bool PostJob(Job::Id id);

  /// PostJobImmediate posts a job to be scheduled immediately.
  ///
  /// @param[in] id  Job id.
  /// @returns True if scheduled successfully.
  bool PostJobImmediate(Job::Id id);

  /// PostJobNext posts a job to run after the current job.
  ///
  /// @param[in] id  Job id.
  /// @returns True if scheduled successfully.
  bool PostJobNext(Job::Id id);

  /// PostJobDelayedMsec posts a job to run delayed by a specific amount.
  ///
  /// @param[in] id  Job id.
  /// @param[in] msec  Time-delay when the job should run.
  /// @returns True if scheduled successfully.
  bool PostJobDelayedMsec(Job::Id id, int msec);

  /// PostJobAtTime posts a job to be scheduled at a specific time.
  ///
  /// @param[in] id  Job id.
  /// @param[in] time  Time when the job should run.
  /// @returns True if scheduled successfully.
  bool PostJobAtTime(Job::Id id, TimePoint time);

  /// PostJobRunAfter posts a job to run after the other job has completed.
  ///
  /// @param[in] id  Job id.
  /// @returns True if scheduled successfully.
  bool PostJobRunAfter(Job::Id id, Job::Id parent);


  /// Reschedule reschedules a job to run after all scheduled jobs.
  ///
  /// @param[in] id  Job id.
  /// @returns True for success or false if jobs was killed.
  bool Reschedule(Job::Id id);

  /// RescheduleDelayedMsec reschedules the job to run after the specified
  /// delay from now.
  ///
  /// @param[in] id  Job id.
  /// @returns True for success or false if jobs was killed.
  bool RescheduleDelayedMsec(Job::Id id, int msec);

  /// RescheduleAtTime reschedules the job to run at a specific time.
  ///
  /// @param[in] id  Job id.
  /// @returns True for success or false if jobs was killed.
  bool RescheduleAtTime(Job::Id id, TimePoint time);

  /// RescheduleAfterJob reeschedule the job to run after the specified job
  /// has finished.
  ///
  /// @param[in] id  Job id.
  /// @param[in] yield  Yield job id.
  /// @param[in] immediate  Schedule yield job immedidately.
  /// @returns True for success or false if jobs was killed.
  bool RescheduleAfterJob(Job::Id id, Job::Id yield, bool inherit_priority = false);

  /// RescheduleAfterEvent reschedules the job to run after the specified
  /// event has occured.
  ///
  /// @param[in] id  Job id.
  /// @returns True for success or false if jobs was killed.
  bool RescheduleAfterEvent(Job::Id, IOEvent::Type, const IOEvent& ioevent);

 private:
  template <typename R, typename... Args>
  Job::Id AllocateJob(R(&&function)(Args...), Args&&...);

  template <typename R, typename C, typename... Args>
  Job::Id AllocateJob(R(C::*)(Args...), C&, Args&&...);

 private:
  void* WorkerRun(WorkerThread* thread);
  void* IOEventThreadRun();

  // Slot
  int AllocateSlot();
  void FreeSlot(int index);

  bool GetWorkerQueueAndLock(WorkerJob& wjob, WorkerThread** pthread, WorkerQueue** pqueue);

  // Queue

  // Returns false if job has been killed or if already queued
  bool QueueWorkerJobLocked(WorkerQueue& queue, WorkerJob& wjob);
  bool QueueWorkerJobLocked(WorkerQueue& queue, WorkerJob& wjob, WorkerJob& wyield);

  WorkerJob* DequeueNextWorkerJobLocked(WorkerQueue& queue, TimePoint&);
  void DequeueWorkerJobLocked(WorkerQueue& queue, WorkerJob& wjob);
  void RemoveWorkerJobLocked(WorkerQueue& queue, WorkerJob& wjob);

  bool WakeWorkerJob(WorkerJob& wjob);
  void WakeBlockedJobsLocked(WorkerQueue& queue, WorkerJob& wjob);

  bool PostWorkerJob(WorkerJob& wjob, TimePoint time);
  bool PostWorkerJob(WorkerJob& wjob, TimePoint time, WorkerJob& wyield);
  bool RescheduleWorkerJob(WorkerJob& wjob, TimePoint time);
  bool RescheduleWorkerJob(WorkerJob& wjob, TimePoint time, WorkerJob& wyield);

  void ReleaseWorkerJob(WorkerJob& wjob);
  void ReleaseWorkerJobLocked(WorkerJob& wjob);

  void DumpQueue(WorkerQueue& queue);

 private:
  bool                    kill_ioevent_handler_;
  Thread*                 ioevent_thread_;
  IOEventHandler          ioevent_handler_;

  bool                    no_threads_;
  bool                    event_thread_;
  unsigned int            max_concurrent_thread_count_;

  mutable std::mutex      job_mutex_;
  unsigned int            job_max_count_;
  uint32_t*               job_alloc_bitmap_;
  WorkerJob*              jobs_;

  mutable std::mutex      thread_mutex_;
  std::list<WorkerThread> threads_;

  mutable std::mutex      queue_mutex_;
  std::list<WorkerQueue>  queues_;
};


} // end of namespace grid

#include "worker/job.h"
#include "worker/queue.h"
#include "worker/thread.h"
#include "worker/worker.h"

#endif  // GRID_UTIL_WORKER_H
