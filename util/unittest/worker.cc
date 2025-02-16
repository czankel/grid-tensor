//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <gtest/gtest.h>

#include <grid/util/worker.h>

using namespace grid;

// Make sure we can create and destroy a worker; also check default thread count
TEST(Worker, SimpleCreate)
{
  Worker worker;
  EXPECT_EQ(Worker::kDefaultConcurrentThreadCount, worker.GetMaxConcurrentThreadCount());
}


// Make sure we can post a static function, member function, and 'bound' fct.
// use locks to make helgrind happy
static std::mutex SimplePost_Mutex;
static volatile bool SimplePost_Complete;

static bool SimplePost_IsComplete()
{
  std::scoped_lock lock(SimplePost_Mutex);
  return SimplePost_Complete;
}

static void SimplePost_Reset()
{
  std::scoped_lock lock(SimplePost_Mutex);
  SimplePost_Complete = false;
}

static bool SimplePost_StaticFunc()
{
  SimplePost_Complete = true;
  return false;
}

static bool SimplePost_StaticFunc2(char a, int i)
{
  SimplePost_Complete = (a == 'a' && i == 1);
  return false;
}

class SimplePost_Class
{
 public:
  SimplePost_Class() : complete_(false) {}
  void reset()
  {
    std::scoped_lock lock(mutex_);
    complete_ = false;
  }
  bool MemberFunc()
  {
    std::scoped_lock lock(mutex_);
    complete_ = true;

    return true;
  }
  bool IsComplete()
  {
    std::scoped_lock lock(mutex_);
    return complete_;
  }
 private:
  volatile bool complete_;
  std::mutex mutex_;
};

TEST(Worker, SimplePost)
{
  Worker worker;
  {
    SimplePost_Reset();
    Job job = worker.Post(SimplePost_StaticFunc);

    // Wait up to 1 second for completion
    int loop;
    for (loop = 0; loop < 100; loop++)
    {
      if (SimplePost_IsComplete())
        break;
      CurrentThread::SleepMsec(10);
    }

    EXPECT_GT(100, loop);
  }

  SimplePost_Class klass;
  {
    klass.reset();
    Job job;
    job = worker.Post(&SimplePost_Class::MemberFunc, klass);

    EXPECT_NE(Job::kInvalid, job.GetId());

    // Wait up to 1 second for completion
    int loop;
    for (loop = 0; loop < 100; loop++)
    {
      if (klass.IsComplete())
        break;
      CurrentThread::SleepMsec(10);
    }
    EXPECT_GT(100, loop);
  }

  // post a 'bound' function with arguments
  {
    SimplePost_Reset();
    Job job = worker.Post(SimplePost_StaticFunc2, 'a', 1);
    // Wait up to 1 second for completion
    int loop;
    for (loop = 0; loop < 100; loop++)
    {
      if (SimplePost_IsComplete())
        break;
      CurrentThread::SleepMsec(10);
    }
    EXPECT_GT(100, loop);
  }
}
