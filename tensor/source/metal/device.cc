//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>

#include <grid/tensor/metal/device.h>

using namespace grid::device;

// FIXME inline?
void CommandEncoder::DispatchThreadgroups(MTL::Size grid_dims, MTL::Size group_dims)
{
  encoder_->dispatchThreadgroups(grid_dims, group_dims);
}

void CommandEncoder::DispatchThreads(MTL::Size grid_dims, MTL::Size group_dims)
{
  encoder_->dispatchThreads(grid_dims, group_dims);
}

void Metal::Wait()
{
  // FIXME
  command_encoder_ = nullptr;
}

Metal::Metal()
{
  auto devices = MTL::CopyAllDevices();
  mtl_device_ = static_cast<MTL::Device*>(devices->object(0)) ?: MTL::CreateSystemDefaultDevice();
  if (!mtl_device_)
    throw std::runtime_error("Failed to load device");

  auto library = NS::String::string(METAL_PATH, NS::UTF8StringEncoding);
  NS::Error* error = nullptr;
  library_ = mtl_device_->newLibrary(library, &error);
  if (error != nullptr)
    throw std::runtime_error("failed to create metal library");

  // TODO: make queues per stream
  queue_ = mtl_device_->newCommandQueue();

  // FIXME here or singleton? command_encoder_(queue_);

#if 0 // FIXME
  NS::AutoreleasePool* pool_;
  pool_ = NS::AutoreleasePool::alloc()->init();
#endif
}


Metal& Metal::GetDevice()
{
  if (g_device_ == nullptr)
    g_device_ = new Metal();

  return *g_device_;
}


MTL::ComputePipelineState* Metal::GetKernel(const std::string& name)
{
  if (auto it = kernels_.find(name); it != kernels_.end())
    return it->second;

  auto function_name = NS::String::string(name.c_str(), NS::ASCIIStringEncoding);
  auto mtl_function = library_->newFunction(function_name);
  if (!mtl_function)
    throw std::runtime_error("Failed to find metal function: " + name);

  NS::Error* error = nullptr;
  auto kernel = mtl_device_->newComputePipelineState(mtl_function, &error);
  if (error)
    throw std::runtime_error(error->localizedDescription()->utf8String());

  kernels_.insert({name, kernel});

  return kernel;
}

CommandEncoder& Metal::Encoder()
{
  if (command_encoder_ == nullptr)
  {
    MTL::CommandBuffer* command_buffer = queue_->commandBufferWithUnretainedReferences();

    if (!command_buffer)
      throw std::runtime_error("failed to create command buffer");

    command_buffer->retain();
    command_encoder_ = std::make_unique<CommandEncoder>(command_buffer);
  }
  return *command_encoder_;
}


#if 0
void Metal::CheckError(MTL::CommandBuffer* cbuf)
{
  if (cbuf->status() == MTL::CommandBufferStatusError) {
    std::ostringstream msg;
    msg << "[METAL] Command buffer execution failed: "
        << cbuf->error()->localizedDescription()->utf8String();
    throw std::runtime_error(msg.str());
  }
}
#endif


/// FIXME: remove: (analysis of mlx)

// Stream stream = default_stream(default_device());


// Stream: has index, queue, and buffer, queue is also in queue_map_ under index, buffer in buffer_map_
//
// Scheduler() : n_active_tasks_(0) {
//   default_streams_.insert({Device::gpu, new_stream(Device::gpu)});
//
// void ::new_stream(Stream stream) {
//   device(stream.device).new_queue(stream.index);
//
// Device::new_queue
//   thread_pool = metal::new_scoped_memory_pool()
//   lock
//   device_->newCommandQueue(MAX_BUFFERS_PER_QUEUE);
//   insert in queue_map_
//
// Device(DeviceType type, int index = 0) : type(type), index(index) {};
//
//
// Any Function (e.g. binary)
//
//   (note: arrays have stream and device)
//   Get stream from output array: out.primitive().stream()
//   Get device from output array: metal::device(stream.device())
//   kernel = device.get_kernel(name)
//   encoder = device.get_command_encoder(s.index)
//   compute_encoder->setComputePipelineState(kernel);
//   compute_encoder->set_input_array, ... ( is basiclly setBuffer..)
//   compute_encoder->dispatchThreads (Device::dispatchThreads)
//   // ??? not commit no task not ??
//
// Device::get_command_encoder(stream.index)
//   (if non-existent) CommandEncoder(get_command_buffer()) and adds it to Device::encoder_map
//
// Device::get_command_buffer(index)
//   lookup in buffer_map_(index)
//   if not found
//     lookup queue_map_(index)
//     create qit->second->commandBufferWithUnretainedReferences();
//     insert in buffer_map_
//
// Device::commit_command_buffer(index)
//   calls buffer->commit() and ->release()
//   removes bit from buffer_map_
//
// Devide::get_command_buffer_ops(index)
//   returns buffer_map_(index).first (count)
//
// Device::increment_command_buffer_ops(int index)
//   increments usage count? called only by std::function<void()> make_task(array arr, bool signal)D
//
// CommandEncoder::CommandEncoder
//   enc = cbuf->computeCommandEncoder(MTL::DispatchTypeConcurrent);
//   enc->retain()
//
// CommandEncoder::dispatchThreads
//   num_dispatches
//   command_buffer->dispatchThreads
//   maybe_split
//
// CommandEncoder::maybe_split
//   if (num_dispatches > MAX_DISPATCHES_PER_ENCODER && !concurrent)
//     enc->endEncoding
//     enc->relaase
//     num_dispatches = 0
//
//     enc = cbuf->computeCommandEncoder(MTL::DispatchTypeConcurrent);
//     enc->retain

// CommandEncoder
//  - dispatchThreadgroups
//  - dispatchThreads
//  - set_input_array
//  - set_output_array
//
//
// Device
//   void new_queue(int index);
//     only used in void new_stream(Stream stream) {
//
//   MTL::CommandBuffer* get_command_buffer(int index);
//   int get_command_buffer_ops(int index);
//   void increment_command_buffer_ops(int index);
//
//   void commit_command_buffer(int index);
//   CommandEncoder& get_command_encoder(int index);
//
//   void end_encoding(int index);
//   void register_library(
//   MTL::Function* get_function(
//   MTL::ComputePipelineState* get_kernel(
//   MTL::ArgumentEncoder* argument_encoder(


// transition.cpp
//
// eval_impl() [called by async_eval() or eval(), eval() does eval().event().wait()]
//
//   std::queue<array> tape;
//
//   std::unordered_map<uint32_t, Event> events;
//
//   Stream stream = default_stream(default_device());
//   (goes through all outputs and assigne stream for the first entry that has one)
//
//   std::unordered_set<uintptr_t> needs_signal;
//   auto synchronizer = array({}, bool_, std::make_shared<Synchronizer>(stream), std::move(outputs));
//   needs_signal.insert(synchronizer.id());
//   events.emplace(stream.index, Event{stream});
//
//   [process all inputs, align synchronization, etc., use dfs]
//   // All inputs are done being processed, process this array
//
//   while (!tape.empty()) {
//     [for gpu]: scheduler::enqueue(stream, metal::make_task(std::move(arr), signal));
//     [cpu much more complicated]
//

// make_task
//   return .. [arr = std::move(arr), signal]() mutable {
//     auto command_buffer = d.get_command_buffer(s.index);
//     d.increment_command_buffer_ops(s.index);
//
//     [[for all input arrays, if they have a event]]
//       input.event().wait();
//
//     arr.primitive().eval_gpu(arr.inputs(), outputs);
//     [[keeps shared ptr to input buffers]]
//
//     if [[signal or too many outstanding ops]]
//       device.end_encoding
//       if [[signal]]
//         command_buffer->encodeSignalEvent(
//         scheduler::notify_new_task(s);
//         command_buffer->addCompletedHandler( ... [...] { notify })
//         d.commit_command_buffer(s.index);
//         d.get_command_buffer(s.index);
//      else
//        command_buffer->addCompletedHandler(... check for errors)
//


grid::device::Metal* grid::device::Metal::g_device_;
