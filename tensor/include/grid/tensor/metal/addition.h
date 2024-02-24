//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

// DO NOT INCLUDE THIS FILE DIRECTLY

#if 0
class metal_adder{
public:

    void init_with_device(MTL::Device* device);
    void prepare_data();
    void send_compute_command();
    void random_number_generator(MTL::Buffer* buffer);
    void encode_add_command(MTL::ComputeCommandEncoder* compute_encoder);
    void verify();


private:
    MTL::Buffer* _A;
    MTL::Buffer* _B;
    MTL::Buffer* _C;

    MTL::Device* _device;
    MTL::CommandQueue* _CommandQueue;
    MTL::ComputePipelineState* _addFunctionPSO;
};

#endif
