//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_UTIL_IOEVENT_H
#define GRID_UTIL_IOEVENT_H

#ifdef __APPLE__
# include <sys/event.h>
# include <sys/types.h>
#else
# include <sys/epoll.h>
# include <sys/eventfd.h>
#endif

namespace grid {

/// IOEvent holds the IO event descriptor. It must be the first entry (head) of any structure
/// using the IOEventHandler.
struct IOEvent
{
  enum Type
  {
    kRead = 1,
    kWrite,
  };

  IOEvent(int descriptor = -1) : descriptor_(descriptor) {}
  int descriptor_;
};

/// IOEventHandler is a wrapper for the system event handlers, such as
/// kevent or epoll.
class IOEventHandler
{
 public:
  static const int kMaxEvents = 1000;

  IOEventHandler();

  bool AddEvent(IOEvent::Type type, IOEvent* event);

  bool RemoveEvent(IOEvent* event);

  IOEvent* WaitForNextEvent();

  void CancelWaitForNextEvent();

 private:
  int     event_fd_;
  int     event_count_;
  int     event_index_;
  IOEvent cancel_event_;

#ifdef __APPLE__
  struct kevent events_[kMaxEvents];
#else
  int     eventfd_;
  struct epoll_event events_[kMaxEvents];
#endif
};


} // end of namespace grid

#endif  // GRID_UTIL_IOEVENT_H
