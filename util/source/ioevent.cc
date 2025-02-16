//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/util/ioevent.h>

namespace grid {

#ifdef __APPLE__

IOEventHandler::IOEventHandler() :
    event_fd_(-1),
    event_count_(0),
    event_index_(0)
{
  event_fd_ = kqueue();

  struct kevent event;
  EV_SET(&event, 0, EVFILT_USER, EV_ADD, 0, 0, 0);
  kevent(event_fd_, &event, 1, &event, 0, 0);
}


bool IOEventHandler::AddEvent(IOEvent::Type type, IOEvent* ioevent)
{
  if (ioevent == 0)
    return false;

  int16_t filter =
      type == IOEvent::kRead ? EVFILT_READ : 0 ||
      type == IOEvent::kWrite ? EVFILT_WRITE : 0;

  struct kevent event;
  EV_SET(&event, ioevent->descriptor_, filter, EV_ADD, 0, 0, ioevent);
  return kevent(event_fd_, &event, 1, &event, 0, 0) == 0;
}


bool IOEventHandler::RemoveEvent(IOEvent* ioevent)
{
  if (ioevent == 0)
    return false;

  struct kevent event;
  EV_SET(&event, ioevent->descriptor_, EVFILT_READ, EV_DELETE, 0, 0, ioevent);
  ioevent->descriptor_ = -1;
  return kevent(event_fd_, &event, 1, 0, 0, 0) == 0;
}


IOEvent* IOEventHandler::WaitForNextEvent()
{
  if (event_index_ == event_count_)
  {
    event_index_ = 0;
    event_count_ = kevent(event_fd_, 0, 0, events_, kMaxEvents, 0);
  }
  if (event_count_ <= 0)
    return 0;

  if (events_[event_index_].udata == 0)
    return 0;

  return static_cast<IOEvent*>(events_[event_index_++].udata);
}


// FIXME: we should check for errors
void IOEventHandler::CancelWaitForNextEvent()
{
  struct kevent event;
  EV_SET(&event, 0, EVFILT_USER, 0, NOTE_TRIGGER, 0, 0);
  kevent(event_fd_, &event, 1, 0, 0, 0);
}

#else // __APPLE__

IOEventHandler::IOEventHandler() :
    event_fd_(-1),
    event_count_(0),
    event_index_(0)
{
  event_fd_ = epoll_create(1);
  eventfd_ = eventfd(0, 0);
  struct epoll_event event = { 0 };
  cancel_event_.descriptor_ = eventfd_;
  event.data.ptr = &cancel_event_;
  event.events = EPOLLIN | EPOLLET;
  epoll_ctl(event_fd_, EPOLL_CTL_ADD, eventfd_, &event);
}


bool IOEventHandler::AddEvent(IOEvent::Type type, IOEvent* ioevent)
{
  if (ioevent == 0)
    return false;

  struct epoll_event event;

  event.events =
      (type == IOEvent::kRead ? EPOLLIN : 0U) |
      (type == IOEvent::kWrite ? EPOLLOUT : 0U);
  event.data.ptr = ioevent;

  return epoll_ctl(event_fd_, EPOLL_CTL_ADD, ioevent->descriptor_, &event) == 0;
}


bool IOEventHandler::RemoveEvent(IOEvent* ioevent)
{
  if (ioevent == 0 || ioevent->descriptor_ < 0)
    return false;
  int descriptor = ioevent->descriptor_;
  ioevent->descriptor_ = -1;

  return epoll_ctl(event_fd_, EPOLL_CTL_DEL, descriptor, 0) == 0;
}


// FIXME: had some run-time problems with this function?
IOEvent* IOEventHandler::WaitForNextEvent()
{
  if (event_count_ < 0)
    return 0;

  int event_count = event_count_;

  if (event_index_ == event_count_)
  {
    event_count = epoll_wait(event_fd_, events_, kMaxEvents, -1);
    event_index_ = 0;
  }
  if (event_count < 0)
    return 0;

  event_count_ = event_count;

  if (events_[event_index_].data.ptr == &cancel_event_)
    return 0;

  return static_cast<IOEvent*>(events_[event_index_++].data.ptr);
}


void IOEventHandler::CancelWaitForNextEvent()
{
  eventfd_write(eventfd_, 1);
}

#endif

} // end of namespace grid
