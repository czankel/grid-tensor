//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/util/worker.h>

#include <cstring>
#include <string>

//#define OVERRIDE_LOG_LEVEL kLogLevelTrace
//#include <grid/util/log.h>
#define TAG "Worker"

#define LOGE(x,...)
#define LOGW(x,...)
#define LOGV(t,f,...) { printf(f,  ##__VA_ARGS__); putchar('\n'); }
#define LOGT(t,f,...) { printf(f,  ##__VA_ARGS__); putchar('\n'); }

namespace grid {

// The Worker implementation uses a single queue per WorkerQueue for all jobs.
// The queue is divided into three sections and three pointers pointing to those sections:
//
//  - ready:     immediately ready to run
//  - scheduled: scheduled for a later time
//  - sleeping:  un-scheduled
//
// When a job is added to the queue, it can be scheduled to run at a specific
// time or with these values:
//
//  - kScheduleImmediate: the job is inserted to the front of the queue. O(1)
//  - kScheduleNormal: the jobs is inserted before all 'scheduled' jobs. O(1)
//  - <time>: the job is inserted to the right position. O(n)
//  - kScheduleSleeping: (sleeping) the jobs is added to the end of the queue O(1)
//
// Any value equal or larger than kScheduleTime is the epoch time when the job
// should be woken up.

static const TimePoint kScheduleImmediate(std::chrono::steady_clock::duration(0));
static const TimePoint kScheduleNormal(SteadyClock::duration(1));
static const TimePoint kScheduleTime(SteadyClock::duration(2));
static const TimePoint kScheduleSleeping(kInfiniteTime);

static __thread WorkerThread* tls_current_thread_ = nullptr;
static __thread uint64_t tls_context_ = 0;

// CurrentJob

__thread Job::Id CurrentJob::tls_current_job_id_ = Job::kInvalid;
//__thread Worker* CurrentJob::tls_current_worker_ = nullptr;
// FIXME __thread WorkerQueue* CurrentJob::tls_current_queue_ = nullptr;

//
// WorkerThread
//

WorkerThread::WorkerThread(const std::string& name, Worker* worker, WorkerQueue* queue)
    : name_(name),
      worker_(worker),
      queue_(queue),
      thread_(nullptr),
      state_(kStateSleeping)
{
}

WorkerThread::WorkerThread(WorkerThread&& other)
    : name_(other.name_),
      worker_(other.worker_),
      queue_(other.queue_),
      thread_(other.thread_),
      state_(kStateSleeping)
{
}


WorkerThread::~WorkerThread()
{
  if (thread_ != nullptr)
  {
    {
      std::scoped_lock lock(queue_->mutex_);
      state_ = kStateKilled;
      queue_->wait_cond_.notify_all();
    }

    thread_->Join();
    delete thread_;
    thread_ = nullptr;
  }
}


void WorkerThread::Start()
{
  LOGT(TAG, "WorkerThread::Start()");
  if (thread_ == nullptr)
  {
    state_ = kStateSleeping;
    thread_ = new Thread(name_.c_str(), std::bind(&Worker::WorkerRun, worker_, this));
    thread_->Start();
  }
}

//
// Worker
//

Worker::Worker(bool no_threads, bool event_thread, unsigned int max_job_count)
    : kill_ioevent_handler_(false),
      ioevent_thread_(nullptr),
      no_threads_(no_threads),
      event_thread_(event_thread),
      max_concurrent_thread_count_(no_threads? 1 : kDefaultConcurrentThreadCount),
      job_max_count_(max_job_count),
      job_alloc_bitmap_(nullptr),
      jobs_(nullptr)
{
  LOGT(TAG, "Worker(nothreads %d, event_thread %d, jobs %d", no_threads, event_thread, max_job_count);

  //assert(job_max_count_ > 0);

  job_alloc_bitmap_ = new uint32_t[(job_max_count_ + 31) / 32];
  //assert(job_alloc_bitmap_ != nullptr);
  memset(job_alloc_bitmap_, 0, (job_max_count_ + 31) / 32);

  jobs_ = new WorkerJob[job_max_count_];
  //assert(jobs_ != nullptr);
  for (unsigned int i = 0; i < job_max_count_; i++)
    new (&jobs_[i]) Job();

  // Create the max number of concurrent threads and queues
  for (unsigned int i = 0; i < max_concurrent_thread_count_; i++)
  {
    std::string name = "worker";
    queues_.push_back(WorkerQueue());

    if (!no_threads || i == 0)
      threads_.push_back(WorkerThread(name, this, &(queues_.back())));
    if (!no_threads)
      threads_.back().Start();
  }

  kill_ioevent_handler_ = false;
  if (event_thread_)
  {
    ioevent_thread_ = new Thread("ioevent_thread", std::bind(&Worker::IOEventThreadRun, this));
    ioevent_thread_->Start();
  }
}


Worker::~Worker()
{
  if (ioevent_thread_ != nullptr)
  {
    kill_ioevent_handler_ = true;
    ioevent_handler_.CancelWaitForNextEvent();
    ioevent_thread_->Join();
    delete ioevent_thread_;
  }

  tls_current_thread_ = nullptr;

  {
    std::scoped_lock lock(thread_mutex_);
    threads_.clear();
  }

  {
    std::scoped_lock lock(queue_mutex_);
    queues_.clear();
  }

  if (job_alloc_bitmap_ != nullptr)
    delete[] job_alloc_bitmap_;
  if (jobs_ != nullptr)
    delete[] jobs_;
}

//
// Worker Public Functions
//

// 'no-thread' Run
bool Worker::Run()
{
  return no_threads_ && WorkerRun(&(threads_.front())) == nullptr;
}


void Worker::Stop()
{
  if (!no_threads_)
  {
    threads_.front().Kill();
  }
}

//
// Worker:: Public Functions
//

bool Worker::SetMaxConcurrentThreadCount(unsigned int count)
{
  //assert(count > 0);

  if (count != max_concurrent_thread_count_)
  {
    max_concurrent_thread_count_ = count;
  }
  return false;
}


unsigned int Worker::GetMaxConcurrentThreadCount()
{
  return max_concurrent_thread_count_;
}


bool Worker::Wake(const Job& job)
{
  Job::Id id = job.GetId();
  if (id == Job::kInvalid)
    return false;

  WorkerJob& wjob = *reinterpret_cast<WorkerJob*>(id);
  WorkerQueue* queue = wjob.queue_;
  if (queue == nullptr)
    return false;

  WakeWorkerJob(wjob);  // omits return value

  return true;
}


void Worker::WakeBlocked(const Job& job)
{
  Job::Id id = job.GetId();
  if (id == Job::kInvalid)
    return;

  WorkerJob& wjob = *reinterpret_cast<WorkerJob*>(id);
  WorkerQueue* queue = wjob.queue_;
  if (queue == nullptr)
    return;

  {
    std::scoped_lock lock(queue->mutex_);
    WakeBlockedJobsLocked(*queue, wjob);
  }

  return;
}


void Worker::KillJob(const Job& job)
{
  Job::Id id = job.GetId();

  if (id != Job::kInvalid)
  {
    LOGV(TAG, "Kill Job %08lx", id);

    WorkerJob& wjob = *reinterpret_cast<WorkerJob*>(id);
    if (wjob.ioevent_scheduled_.load())
    {
      ioevent_handler_.RemoveEvent(&wjob.ioevent_);
      wjob.ioevent_scheduled_ = false;
    }

    // modify reschedule, once set to kKill, it cannot change
    wjob.reschedule_ = WorkerJob::kKill;

    WorkerQueue* queue = wjob.queue_;
    if (queue != nullptr)
    {
      std::scoped_lock lock(queue->mutex_);
      if (wjob.is_queued_)
        DequeueWorkerJobLocked(*queue, wjob);
      ReleaseWorkerJobLocked(wjob);
    }
  }
}


bool Worker::IsRescheduled(const Job& job)
{
  Job::Id id = job.GetId();
  return id != Job::kInvalid &&
    reinterpret_cast<WorkerJob*>(id)->reschedule_.load() == WorkerJob::kAgain;
}


bool Worker::IsRescheduledWaiting(const Job& job)
{
  Job::Id id = job.GetId();
  if (id == Job::kInvalid)
    return false;

  TimePoint now = SteadyClock::now();
  //uint64_t monotonic_time = Time::GetMonotonicTime().GetMilliseconds();
  WorkerJob* wjob = reinterpret_cast<WorkerJob*>(id);

  return wjob->yield_ != nullptr || wjob->scheduled_time_ > now;;
}


// TODO implement Worker::WaitForJob
bool Worker::WaitForJob(const Job& job)
{
  return false;
}


// TODO implement Worker::CancelWaitForJob
void Worker::CancelWaitForJob(const Job& job)
{
}

//
// Worker Protected Functions
//

Job::Id Worker::AllocateJob(size_t size)
{
  if (size > WorkerJob::kMaxFunctorSize)
    throw std::runtime_error("size of job function too large");

  int slot = AllocateSlot();
  if (slot < 0)
    return Job::kInvalid;

  WorkerJob& wjob = jobs_[slot];

  wjob.refcount_ = 1;
  wjob.context_ = tls_context_;

  // reset important fields
  wjob.ioevent_ = IOEvent();
  wjob.ioevent_scheduled_ = false;

  wjob.next_ = nullptr;
  wjob.prev_ = nullptr;

  wjob.block_ = nullptr;
  wjob.yield_ = nullptr;

  wjob.worker_ = this;
  wjob.queue_ = nullptr;
  wjob.is_queued_ = false;

  wjob.result_ = false;
  wjob.running_ = false;

  wjob.reschedule_ = WorkerJob::kOnce;
  wjob.woken_ = false;

  return reinterpret_cast<Job::Id>(&wjob);
}


void* Worker::GetFunctionPointer(Job::Id id)
{
  if (id == Job::kInvalid)  // FIXME throw?
    throw std::runtime_error("function pointer of an invalid job requested");

  return reinterpret_cast<WorkerJob*>(id)->function_buffer_;
}


void Worker::AddRefJob(Job::Id id)
{
  if (id != Job::kInvalid)
    ++reinterpret_cast<WorkerJob*>(id)->refcount_;
}


void Worker::ReleaseJob(Job::Id id)
{
  if (id != Job::kInvalid)
    ReleaseWorkerJob(*reinterpret_cast<WorkerJob*>(id));
}


void Worker::SetContext(Job::Id id, uint64_t context)
{
  if (id != Job::kInvalid)
    reinterpret_cast<WorkerJob*>(id)->context_ = context;
}


uint64_t Worker::GetContext(Job::Id id) const
{
  return id != Job::kInvalid ? reinterpret_cast<WorkerJob*>(id)->context_ : 0;
}


Job::Status Worker::GetJobStatus(Job::Id id) const
{
  if (id == Job::kInvalid)
    return Job::kInvalid;

  WorkerJob* wjob = reinterpret_cast<WorkerJob*>(id);
  return wjob->is_queued_ ? Job::kWaiting : (
           wjob->reschedule_ != WorkerJob::kKill && wjob->running_ ? Job::kRunning : (
             wjob->result_ ? Job::kDone : Job::kError));
}



bool Worker::NeedsReschedule(Job::Id id) const
{
  if (id == Job::kInvalid)
    return false;

  WorkerQueue* queue = reinterpret_cast<WorkerJob*>(id)->queue_;
  if (queue == nullptr)
    return false;

  return queue->needs_reschedule_.load();
}

// Posting Jobs

// Note that for the follwoing, PostWorkerJob handles 'invalid' jobs
bool Worker::PostJob(Job::Id id)
{
  if (id == Job::kInvalid)
    return false;

  return PostWorkerJob(*reinterpret_cast<WorkerJob*>(id), kScheduleNormal);
}


bool Worker::PostJobNext(Job::Id id)
{
  if (id == Job::kInvalid)
    return false;

  return PostWorkerJob(*reinterpret_cast<WorkerJob*>(id), kScheduleNormal);
}


bool Worker::PostJobImmediate(Job::Id id)
{
  if (id == Job::kInvalid)
    return false;

  return PostWorkerJob(*reinterpret_cast<WorkerJob*>(id), kScheduleImmediate);
}


bool Worker::PostJobDelayedMsec(Job::Id id, int msec)
{
  if (id == Job::kInvalid)
    return false;

  // FIXME uint64_t monotonic_time = Time::GetMonotonicTime().GetMilliseconds();
  return PostWorkerJob(*reinterpret_cast<WorkerJob*>(id),
                       SteadyClock::now() + std::chrono::milliseconds(msec));
}


bool Worker::PostJobAtTime(Job::Id id, TimePoint time)
{
  if (id == Job::kInvalid)
    return false;

  return PostWorkerJob(*reinterpret_cast<WorkerJob*>(id), time);
}


bool Worker::PostJobRunAfter(Job::Id id, Job::Id yield)
{
  if (id == Job::kInvalid || yield == Job::kInvalid)
    return false;

  return PostWorkerJob(*reinterpret_cast<WorkerJob*>(id), kScheduleNormal,
                       *reinterpret_cast<WorkerJob*>(yield));
}


// Note that the current job is not in the queue, so we can change its fields
bool Worker::Reschedule(Job::Id id)
{
  return id != Job::kInvalid &&
    RescheduleWorkerJob(*reinterpret_cast<WorkerJob*>(id), kScheduleNormal);
}


bool Worker::RescheduleDelayedMsec(Job::Id id, int msec)
{
  if (id == Job::kInvalid)
    return false;

  return RescheduleWorkerJob(*reinterpret_cast<WorkerJob*>(id),
                             SteadyClock::now() + std::chrono::milliseconds(msec));
}


bool Worker::RescheduleAtTime(Job::Id id, TimePoint time)
{
  if (id == Job::kInvalid)
    return false;
  return RescheduleWorkerJob(*reinterpret_cast<WorkerJob*>(id), time);
}


bool Worker::RescheduleAfterJob(Job::Id id, Job::Id yield, bool immediate)
{
  if (id == Job::kInvalid || yield == Job::kInvalid)
    return false;

  // note that if yield is invalid, it will just reschedule the job normally
  TimePoint time = immediate ? kScheduleImmediate : kScheduleNormal;
  return RescheduleWorkerJob(*reinterpret_cast<WorkerJob*>(id), time,
                             *reinterpret_cast<WorkerJob*>(yield));
}


bool Worker::RescheduleAfterEvent(Job::Id id, IOEvent::Type type, const IOEvent& ioevent)
{
  if (id == Job::kInvalid)
    return false;

  WorkerJob* wjob = reinterpret_cast<WorkerJob*>(id);
  if (wjob->reschedule_ == WorkerJob::kKill)
    return false;

  // Only allow to schedule a single event
  bool expect_false = false;
  if (wjob->ioevent_scheduled_.compare_exchange_strong(expect_false, true))
  {
    wjob->ioevent_ = ioevent;
    ioevent_handler_.AddEvent(type, &wjob->ioevent_);
  }

  return RescheduleWorkerJob(*wjob, kInfiniteTime);
}


//
// Worker Private Functions
//

// WorkerRun runs until the thread is killed.
//
// Return if no_thread_ and job queue is empty
void* Worker::WorkerRun(WorkerThread* thread)
{
  LOGT(TAG, "Worker::WorkerRun");

  tls_current_thread_ = thread;
  WorkerQueue& queue = *(thread->queue_); // FIXME

  while (thread->state_ != WorkerThread::kStateKilled)
  {
#if 1
    //if (log::MaxLevel >= log::kLogLevelVerbose)
      DumpQueue(queue); // debugging
#endif

    // FIXME duration??
    TimePoint timeout;
    WorkerJob* wjob;

    {
      std::unique_lock lock(queue.mutex_);
      wjob = DequeueNextWorkerJobLocked(queue, timeout);

      if (thread->state_ == WorkerThread::kStateKilled)
        break;

      if (wjob == nullptr)
      {
        if (no_threads_ && queue.ready_ == 0)
          break;
        if (timeout != kScheduleImmediate && timeout != kScheduleSleeping)
          queue.wait_cond_.wait_until(lock, timeout);
        else
          queue.wait_cond_.wait(lock);
        continue;
      }
    }

    if (wjob == nullptr)
      return (void*)-1;

    wjob->next_ = nullptr;
    wjob->prev_ = nullptr;
    wjob->scheduled_time_ = kScheduleNormal;

    SetCurrentJob(reinterpret_cast<Job::Id>(wjob));

    while (thread->state_ == WorkerThread::kStateRunning)
    {
      // set running in the loop to ensure we won't get a false-positive
      // 'killed' job status. (might get a false-positive run status)
      WorkerJob::Reschedule reschedule;
      do
      {
        reschedule = wjob->reschedule_.load();
        wjob->running_ = reschedule != WorkerJob::kKill;
      }
      while (reschedule != WorkerJob::kKill &&
             !wjob->reschedule_.compare_exchange_strong(reschedule, WorkerJob::kOnce));

      if (reschedule != WorkerJob::kKill)
      {
#if 0
        // set context for log
        Log* log = Log::GetLog();
        log->SetContext(wjob->context_);
#endif
        tls_context_ = wjob->context_;

        // execute job; returns false to quit (kill) job
        //const FunctionT<bool()>& functor =
        //  reinterpret_cast<FunctionT<bool()>&>(*wjob-unction_buffer_);
        const std::function<bool()>& functor =
          reinterpret_cast<std::function<bool()>&>(*wjob->function_buffer_);

        wjob->result_ = functor();
        wjob->running_ = false;
        if (!wjob->result_)
          wjob->reschedule_ = WorkerJob::kKill;

        // TODO if wjob.yield_ != nullptr, we can immediately run the yield_ job
        reschedule = wjob->reschedule_.load();
      }

      if (reschedule == WorkerJob::kAgain)
      {
        if (queue.needs_reschedule_.load())
        {
          std::scoped_lock lock (queue.mutex_);
          // don't add to main queue if already added to a yielding job
          if (wjob->yield_ == nullptr && !QueueWorkerJobLocked(queue, *wjob)) {
            ReleaseWorkerJobLocked(*wjob); }
          break;
        }
        // else: 'continue'
      }
      else
      {
        // note that this will wake up blocked jobs
        ReleaseWorkerJob(*wjob);
        break;
      }
    }
    SetCurrentJob(Job::kInvalid);
  }

  tls_current_thread_ = nullptr;

  return nullptr;
}

// IO Event Thread

void* Worker::IOEventThreadRun()
{
  while (!kill_ioevent_handler_)
  {
    IOEvent* ioevent = ioevent_handler_.WaitForNextEvent();
    if (ioevent == nullptr)
      continue;

    // ioevent handler is 'level triggered'
    ioevent_handler_.RemoveEvent(ioevent);

    WorkerJob& wjob = *reinterpret_cast<WorkerJob*>(ioevent);
    wjob.ioevent_scheduled_ = false;
    WakeWorkerJob(wjob);  // omits return value
  }
  return 0;
}

//
// Slot (Bitmap Allocator)
//

// Find a free slot from the allocation bitmap
int Worker::AllocateSlot()
{
  std::scoped_lock lock(job_mutex_);

  for (unsigned int i = 0; i < job_max_count_ / 32; i++)
  {
    if (job_alloc_bitmap_[i] != ~0U)
    {
      uint32_t b = job_alloc_bitmap_[i];
      int index = 0;
      b = ~b & (b+1);

      if ((b & 0xffff0000) != 0)
        index += 16;
      if ((b & 0xff00ff00) != 0)
        index += 8;
      if ((b & 0xf0f0f0f0) != 0)
        index += 4;
      if ((b & 0xcccccccc) != 0)
        index += 2;
      if ((b & 0xaaaaaaaa) != 0)
        index += 1;
      job_alloc_bitmap_[i] |= 1 << index;
      return index + i * 32;
    }
  }
  // no space left
  return -1;
}


void Worker::FreeSlot(int index)
{
  std::scoped_lock lock(job_mutex_);
  job_alloc_bitmap_[index/32] &= ~ (1 << (index & 31));
}


// TODO: optimization possible, e.g. which queue to pick
bool Worker::GetWorkerQueueAndLock(WorkerJob& wjob, WorkerThread** pthread, WorkerQueue** pqueue)
{
  WorkerThread* thread = tls_current_thread_;

  std::scoped_lock thread_lock(thread_mutex_);

  if (thread == nullptr)
    thread = &threads_.front();

  if (thread == nullptr)
  {
    queues_.push_back(WorkerQueue());
    threads_.push_back(WorkerThread("baseworker", this, &queues_.back()));
    thread = &threads_.back();
    // FIXME: not setting tls_current_thread_?
  }

  wjob.queue_ = thread->queue_;

  thread->SetRunning();

  // Note that we keep the queue locked but let the Thread lock expire
  thread->queue_->mutex_.lock();

  if (pthread != nullptr)
    *pthread = thread;
  if (pqueue != nullptr)
    *pqueue = thread->queue_;

  return true;
}

//
// Queue
//

inline static WorkerJob* ThisFromNext(WorkerJob** next)
{
  return (WorkerJob*)((uintptr_t)next - (uintptr_t)(&(((WorkerJob*)0))->next_));
}


// Managing the Queue (queue has to be locked for these functions)
//
// Notes
//  * For performance reasons, most fields of the WorkerJob are only
//    initialized when they leave the locked area.
//    These fields are  next_, prev_, reschedule_
//
//  * scheduled_ and sleeping_ point to the pointers for the first
//    scheduled or sleeping job. This is usually the next_ pointer of the
//    last job of the previous category, or ready_ if no job is in any
//    of the previous categories.
//
//  * The 'locked' versions can assume that queue and job are non-null.
//
// Returns the pointer to the job where next points to &job->next_
bool Worker::QueueWorkerJobLocked(WorkerQueue& queue, WorkerJob& wjob)
{
  LOGT(TAG, "QueueWorkerJobLocked wjob %p sched %lld", &wjob, wjob.scheduled_time_.time_since_epoch().count());

  if (wjob.reschedule_.load() == WorkerJob::kKill)
    return false;

  if (wjob.is_queued_ || wjob.yield_ != nullptr)
  {
    LOGE(TAG, "Job already queued!");
    return false;
  }

  // handle case when job was woken up
  if (wjob.woken_.load() && wjob.scheduled_time_ >= kScheduleTime)
    wjob.scheduled_time_ = kScheduleNormal;
  wjob.woken_ = false;

  TimePoint time = wjob.scheduled_time_;
  if (time == kInfiniteTime)
  {
    WorkerJob** sleeping = queue.sleeping_;
    wjob.prev_ = sleeping != &queue.ready_ ? ThisFromNext(sleeping) : nullptr;
    wjob.next_ = *sleeping;
    if (*sleeping != nullptr)
      (*sleeping)->prev_ = &wjob;
    *sleeping = &wjob;   // note: sets ready_ if sleeping pointed to it
  }

  // insert job before the job that is scheduled later (note O(n)!)
  else if (time > kScheduleTime)
  {
    WorkerJob* prev = queue.sleeping_ != &queue.ready_ ?  ThisFromNext(queue.sleeping_) : nullptr;
    WorkerJob* next = *queue.sleeping_;
    while (prev != nullptr && time < prev->scheduled_time_ )
      next = prev, prev = prev->prev_;
    wjob.prev_ = prev;
    wjob.next_ = next;

    // next == 0 -> prev: last scheduled job or nullptr if none is scheduled
    if (next != nullptr)
      next->prev_ = &wjob;

    if (next == *queue.sleeping_)
      queue.sleeping_ = &wjob.next_;

    // prev == 0 -> no jobs ready, scheduled_ points to ready_
    if (prev != nullptr)
      prev->next_ = &wjob;
    else
      queue.ready_ = &wjob;
  }

  // insert job at ready_ before any other jobs
  else if (time == kScheduleImmediate)
  {
    // insert job before all other jobs
    wjob.prev_ = nullptr;
    wjob.next_ = queue.ready_;
    queue.ready_ = &wjob;
    if (wjob.next_ != nullptr)
      wjob.next_->prev_ = &wjob;

    // push scheduled and sleeping pointers, if necessary
    if (queue.scheduled_ == &queue.ready_)
      queue.scheduled_ = &wjob.next_;
    if (queue.sleeping_ == &queue.ready_)
      queue.sleeping_ = &wjob.next_;
  }

  // insert job at the end of any ready job and before any 'scheduled' job
  else /* kScheduleNormal */
  {
    WorkerJob** scheduled = queue.scheduled_;
    wjob.prev_ = scheduled != &queue.ready_ ? ThisFromNext(scheduled) : nullptr;
    wjob.next_ = *scheduled;
    if (*scheduled != nullptr)
      (*scheduled)->prev_ = &wjob;
    *scheduled = &wjob;  // updates ready_ if scheduled_ pointed there

    // push scheduled and sleeping pointers
    queue.scheduled_ = &wjob.next_;
    if (queue.sleeping_ == scheduled)
      queue.sleeping_ = &wjob.next_;
  }
  wjob.is_queued_ = true;
  return true;
}

bool Worker::QueueWorkerJobLocked(WorkerQueue& queue, WorkerJob& wjob, WorkerJob& wyield)
{
  LOGT(TAG, "QueueWorkerJobLocked wjob %p sched %lld yield %p", &wjob, wjob.scheduled_time_.time_since_epoch().count(), &wyield);

  if (wjob.reschedule_.load() == WorkerJob::kKill)
    return false;

  if (wjob.is_queued_ || wjob.yield_ != nullptr)
  {
    LOGE(TAG, "Job already queued!");
    return false;
  }

  // handle case when job was woken up
  if (wjob.woken_.load() && wjob.scheduled_time_ >= kScheduleTime)
    wjob.scheduled_time_ = kScheduleNormal;
  wjob.woken_ = false;

  // insert the job to the blocked queue of the yielding job
  wjob.next_ = wyield.block_;
  wjob.prev_ = nullptr;

  if (wjob.next_ != nullptr)
    wjob.next_->prev_ = &wjob;

  wyield.block_ = &wjob;
  wjob.yield_ = &wyield;

  wjob.is_queued_ = true;

  return true;
}


// Dequeue (can be from the main queue or from a 'block_' queue)
void Worker::DequeueWorkerJobLocked(WorkerQueue& queue, WorkerJob& wjob)
{
  LOGT(TAG, "DequeueWorkerJobLocked %p yield %p", &wjob, &wjob.yield_);
  WorkerJob* prev = wjob.prev_;
  WorkerJob* next = wjob.next_;

  if (prev != nullptr)
    prev->next_ = wjob.next_;
  if (next != nullptr)
    next->prev_ = wjob.prev_;

  // adjust main queue pointers if removed from there
  if (wjob.yield_ == nullptr)
  {
    if (queue.sleeping_ == &wjob.next_)
      queue.sleeping_ = prev != nullptr ? &prev->next_ : &queue.ready_;
    if (queue.scheduled_ == &wjob.next_)
      queue.scheduled_ = prev != nullptr ? &prev->next_ : &queue.ready_;
    if (queue.ready_ == &wjob)
      queue.ready_ = next;
  }
  else
  {
    if (wjob.yield_->block_ == &wjob)
      wjob.yield_->block_ = next;
    wjob.yield_ = nullptr;
  }
  wjob.is_queued_ = false;
}


void Worker::RemoveWorkerJobLocked(WorkerQueue& queue, WorkerJob& wjob)
{
  LOGT(TAG, "RemoveWorkerJobLocked(wjob %p)", &wjob);

  DequeueWorkerJobLocked(queue, wjob);
  WakeBlockedJobsLocked(queue, wjob);

  int job_index = &wjob - jobs_;
  FreeSlot(job_index);
}


void Worker::DumpQueue(WorkerQueue& queue)
{
  std::scoped_lock lock(queue.mutex_);
#if 1
  //if (log::MaxLevel >= log::kLogLevelDebug)
  {
    LOGV(TAG, "-------------------");
    WorkerJob* wjob;
    if (queue.scheduled_ == &queue.ready_)
        LOGV(TAG, "-- scheduled --");
    if (queue.sleeping_ == &queue.ready_)
        LOGV(TAG, "-- sleeping --");
    for (wjob = queue.ready_; wjob != nullptr; wjob = wjob->next_)
    {
      LOGV(TAG, "Job %p time %lld next %p yield %p block %p", wjob,
                wjob->scheduled_time_.time_since_epoch().count(), wjob->next_, wjob->yield_, wjob->block_);
      if (wjob->next_ == *queue.scheduled_)
        LOGV(TAG, "-- scheduled --");
      if (wjob->next_ == *queue.sleeping_)
        LOGV(TAG, "-- sleeping --");
    }
  }
#endif
}

WorkerJob*
Worker::DequeueNextWorkerJobLocked(WorkerQueue& queue, TimePoint& timeout)
{
  // we are 'rescheduling' right now, so clear flag
  queue.needs_reschedule_ = false;

again:
  // scheduled_ can never be nullptr, just security check
  WorkerJob* wjob = queue.scheduled_ != nullptr ? *queue.scheduled_ : nullptr;
  TimePoint next_scheduled_time = wjob != nullptr ? wjob->scheduled_time_ : kScheduleImmediate; // FIXME

  if (wjob == nullptr || wjob->scheduled_time_ > SteadyClock::now())
    wjob = (wjob != queue.ready_ ? queue.ready_ : nullptr);

  if (wjob != nullptr)
  {
    DequeueWorkerJobLocked(queue, *wjob);
    if (wjob->reschedule_.load() == WorkerJob::kKill && --wjob->refcount_ == 0)
    {
      RemoveWorkerJobLocked(queue, *wjob);
      goto again;
    }
  }

  timeout = next_scheduled_time;

  return wjob;
}

//
// Wake
//

// returns false if job has been killed
bool Worker::WakeWorkerJob(WorkerJob& wjob)
{
  LOGT(TAG, "WakeWorkerJob %p", &wjob);
  bool woken = false;

  std::scoped_lock lock(wjob.queue_->mutex_);
  wjob.woken_ = true;

  // is job is already queued, we need to re-queue it
  if (wjob.is_queued_)
  {
    DequeueWorkerJobLocked(*wjob.queue_, wjob);
    wjob.scheduled_time_ = kScheduleNormal;

    if (QueueWorkerJobLocked(*wjob.queue_, wjob))
    {
      wjob.queue_->needs_reschedule_ = true;
      wjob.queue_->wait_cond_.notify_one();
      woken = true;
    }
    else
    {
      ReleaseWorkerJobLocked(wjob);
    }
  }

  return woken;
}


// TODO: wblocked might be in a different queue that's not been locked
void Worker::WakeBlockedJobsLocked(WorkerQueue& queue, WorkerJob& wjob)
{
  LOGT(TAG, "WakeBlockedJobsLocked (wjob %p)", &wjob);

  // 'disconnect' the list from the blocking job and walk the list
  WorkerJob* next = wjob.block_;
  wjob.block_ = nullptr;
  while (next != nullptr)
  {
    WorkerJob* wblocked = next;
    next = wblocked->next_;

    wblocked->yield_ = nullptr;
    wblocked->is_queued_ = false;

    if (wblocked->reschedule_.load() != WorkerJob::kKill)
      QueueWorkerJobLocked(queue, *wblocked);
    else if (--wjob.refcount_ == 0)
      RemoveWorkerJobLocked(queue, *wblocked);
  }
}

//
// Post Job
//

bool Worker::PostWorkerJob(WorkerJob& wjob, TimePoint time)
{
  WorkerQueue* queue = nullptr;

  wjob.reschedule_ = WorkerJob::kOnce;
  wjob.scheduled_time_ = time;
  WorkerThread* thread;

  //// FIXME: manage Group (rename to Queue?)
  if (GetWorkerQueueAndLock(wjob, &thread, &queue))
  {
    QueueWorkerJobLocked(*queue, wjob);
    queue->wait_cond_.notify_one();
    queue->mutex_.unlock();
  }

  return queue != nullptr;
}


// Note that when the yield job is not alive, this job will automatically be
// scheduled after the yield job is released.
// TODO: yield.queue is not protected
bool Worker::PostWorkerJob(WorkerJob& wjob, TimePoint time, WorkerJob& wyield)
{
  WorkerQueue* queue = nullptr;

  wjob.reschedule_ = WorkerJob::kOnce;
  wjob.scheduled_time_ = time;

  queue = wyield.queue_;
  if (queue != nullptr)
  {
    std::scoped_lock lock(queue->mutex_);
    QueueWorkerJobLocked(*queue, wjob, wyield);
  }
  return queue != nullptr;
}

//
// Reschedule
//

bool
Worker::RescheduleWorkerJob(WorkerJob& wjob, TimePoint time)
{
  LOGT(TAG, "RescheduleWorkerJob %p scheduled %lld", &wjob, time.time_since_epoch().count());

  if (wjob.queue_ == nullptr)
    return false;

  WorkerQueue& queue = *wjob.queue_;

  // set reschedule flag to kAgain unless it has been killed
  WorkerJob::Reschedule reschedule;
  wjob.scheduled_time_ = time;
  do
  {
    reschedule = wjob.reschedule_.load();
    if (reschedule == WorkerJob::kKill)
      return false;
  }
  while (!wjob.reschedule_.compare_exchange_strong(reschedule, WorkerJob::kAgain));

  /// don't force a reschedule if the job is currently running and not
  /// scheduled for a later time. (job is normally running at this point)
  if (wjob.is_queued_ || time > kScheduleNormal)
    queue.needs_reschedule_ = true;

  return true;
}


bool Worker::RescheduleWorkerJob(WorkerJob& wjob, TimePoint time, WorkerJob& wyield)
{
  LOGT(TAG, "RescheduleWorkerJob %p yield %p scheduled %lld", &wjob, &wyield, time.time_since_epoch().count());

  if (wjob.queue_ == nullptr)
    return false;

  WorkerQueue& queue = *wjob.queue_;
  wjob.scheduled_time_ = time;

  // set reschedule flag to kAgain unless it has been killed
  WorkerJob::Reschedule reschedule;
  do
  {
    reschedule = wjob.reschedule_.load();
    if (reschedule == WorkerJob::kKill)
      return false;
  }
  while (!wjob.reschedule_.compare_exchange_strong(reschedule, WorkerJob::kAgain));

  // TODO: check for loop with yielding jobs
  // TODO: yield could be in a different queue which will break!

  // Note about locking: the lock protects all queues (main queue and
  // each job blocked queue), so even if the yielding job gets killed
  // while inside the locked area below, the lock prohibits that the
  // added job will be lost as the destroying of the yielding job will
  // wait for the lock before releasing all queued jobs.
  {
    std::scoped_lock lock (queue.mutex_);
    if (wyield.reschedule_.load() != WorkerJob::kKill)
    {
      wjob.prev_ = nullptr;
      wjob.next_ = wyield.block_;
      wyield.block_ = &wjob;
      if (wjob.next_ != nullptr)
        wjob.next_->prev_ = &wjob;
      wjob.yield_ = &wyield;
      queue.needs_reschedule_ = true;
      wjob.is_queued_ = true;    // note: queued in the 'yield' job queue
    }
  }

  return true;
}

//
// Release
//

void Worker::ReleaseWorkerJobLocked(WorkerJob& wjob)
{
  if (wjob.refcount_.load() == 0)
    LOGW(TAG, "Job already removed!");

  if (wjob.queue_ != nullptr && --wjob.refcount_ == 0)
    RemoveWorkerJobLocked(*wjob.queue_, wjob);
}


// note that RemoveWorkerJobLocked wakes up any blocked jobs
void Worker::ReleaseWorkerJob(WorkerJob& wjob)
{
  if (wjob.refcount_.load() == 0)
  {
    LOGW(TAG, "Job already removed!");
    return;
  }

  if (wjob.queue_ != nullptr && --wjob.refcount_ == 0)
  {
    std::scoped_lock lock(wjob.queue_->mutex_);
    RemoveWorkerJobLocked(*wjob.queue_, wjob);
  }
}


} // end of namespace grid
