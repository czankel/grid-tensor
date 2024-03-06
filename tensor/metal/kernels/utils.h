//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef TENSOR_METAL_KERNELS_UTILS_H
#define TENSOR_METAL_KERNELS_UTILS_H

// FIXME: doesn't this alrady exist somewhere?
#define stringify_(s) #s
#define stringify(s) stringify_(s)

// helpers to return the number of arguments
#define PP_NARG(...)  PP_NARG_(__VA_ARGS__,PP_RSEQ_N())
#define PP_NARG_(...) PP_ARG_N(__VA_ARGS__)
#define PP_ARG_N( _1, _2, _3, _4, _5, _6, _7, _8, _9,_10, \
                 _11,_12,_13,_14,_15,_16,_17,_18,_19,_20, \
                 _21,_22,_23,_24,_25,_26,_27,_28,_29,_30, \
                 _31,_32,_33,_34,_35,_36,_37,_38,_39,_40, \
                 _41,_42,_43,_44,_45,_46,_47,_48,_49,_50, \
                 _51,_52,_53,_54,_55,_56,_57,_58,_59,_60, \
                 _61,_62,_63,N,...) N
#define PP_RSEQ_N() 63,62,61,60, \
                    59,58,57,56,55,54,53,52,51,50, \
                    49,48,47,46,45,44,43,42,41,40, \
                    39,38,37,36,35,34,33,32,31,30, \
                    29,28,52,52,25,24,23,22,21,20, \
                    19,18,17,16,15,14,13,12,11,10, \
                     9, 8, 7, 6, 5, 4, 3, 2, 1, 0

// helper to extract the list of arguments passed to the macro. 
#define ARG_LIST(...) __VA_ARGS__

// Expand the argument type
#define EXPAND_TYPE_1(FUNC, R, O, T, ...)   FUNC(R, O, T)
#define EXPAND_TYPE_2(FUNC, R, O, T, ...)   FUNC(R, O, T) EXPAND_TYPE_1(FUNC, R, O, __VA_ARGS__)
#define EXPAND_TYPE_3(FUNC, R, O, T, ...)   FUNC(R, O, T) EXPAND_TYPE_2(FUNC, R, O, __VA_ARGS__)
#define EXPAND_TYPE_4(FUNC, R, O, T, ...)   FUNC(R, O, T) EXPAND_TYPE_3(FUNC, R, O, __VA_ARGS__)
#define EXPAND_TYPE_5(FUNC, R, O, T, ...)   FUNC(R, O, T) EXPAND_TYPE_4(FUNC, R, O, __VA_ARGS__)
#define EXPAND_TYPE_6(FUNC, R, O, T, ...)   FUNC(R, O, T) EXPAND_TYPE_5(FUNC, R, O, __VA_ARGS__)
#define EXPAND_TYPE_7(FUNC, R, O, T, ...)   FUNC(R, O, T) EXPAND_TYPE_6(FUNC, R, O, __VA_ARGS__)
#define EXPAND_TYPE_8(FUNC, R, O, T, ...)   FUNC(R, O, T) EXPAND_TYPE_7(FUNC, R, O, __VA_ARGS__)
#define EXPAND_TYPE_9(FUNC, R, O, T, ...)   FUNC(R, O, T) EXPAND_TYPE_8(FUNC, R, O, __VA_ARGS__)
#define EXPAND_TYPE_10(FUNC, R, O, T, ...)  FUNC(R, O, T) EXPAND_TYPE_9(FUNC, R, O, __VA_ARGS__)
#define EXPAND_TYPE_11(FUNC, R, O, T, ...)  FUNC(R, O, T) EXPAND_TYPE_10(FUNC, R, O, __VA_ARGS__)
#define EXPAND_TYPE_12(FUNC, R, O, T, ...)  FUNC(R, O, T) EXPAND_TYPE_11(FUNC, R, O, __VA_ARGS__)
#define EXPAND_TYPE_13(FUNC, R, O, T, ...)  FUNC(R, O, T) EXPAND_TYPE_12(FUNC, R, O, __VA_ARGS__)
#define EXPAND_TYPE_14(FUNC, R, O, T, ...)  FUNC(R, O, T) EXPAND_TYPE_13(FUNC, R, O, __VA_ARGS__)
#define EXPAND_TYPE_15(FUNC, R, O, T, ...)  FUNC(R, O, T) EXPAND_TYPE_14(FUNC, R, O, __VA_ARGS__)
#define EXPAND_TYPE_16(FUNC, R, O, T, ...)  FUNC(R, O, T) EXPAND_TYPE_15(FUNC, R, O, __VA_ARGS__)
#define EXPAND_TYPE_17(FUNC, R, O, T, ...)  FUNC(R, O, T) EXPAND_TYPE_16(FUNC, R, O, __VA_ARGS__)
#define EXPAND_TYPE_18(FUNC, R, O, T, ...)  FUNC(R, O, T) EXPAND_TYPE_17(FUNC, R, O, __VA_ARGS__)
#define EXPAND_TYPE_19(FUNC, R, O, T, ...)  FUNC(R, O, T) EXPAND_TYPE_18(FUNC, R, O, __VA_ARGS__)

#define EXPAND_TYPE_(NARGS, ...)            EXPAND_TYPE_ ## NARGS(__VA_ARGS__)
#define EXPAND_TYPE(NARGS, FUNC, R, O, T)   EXPAND_TYPE_(NARGS, FUNC, R, O, ARG_LIST T)

// Expand the operators
#define EXPAND_OP_1(FUNC, R, T, O, ...)     EXPAND_TYPE(PP_NARG T, FUNC, R, O, T)
#define EXPAND_OP_2(FUNC, R, T, O, ...)     EXPAND_TYPE(PP_NARG T, FUNC, R, O, T) EXPAND_OP_1(FUNC, R, T, __VA_ARGS__)
#define EXPAND_OP_3(FUNC, R, T, O, ...)     EXPAND_TYPE(PP_NARG T, FUNC, R, O, T) EXPAND_OP_2(FUNC, R, T, __VA_ARGS__)
#define EXPAND_OP_4(FUNC, R, T, O, ...)     EXPAND_TYPE(PP_NARG T, FUNC, R, O, T) EXPAND_OP_3(FUNC, R, T, __VA_ARGS__)
#define EXPAND_OP_5(FUNC, R, T, O, ...)     EXPAND_TYPE(PP_NARG T, FUNC, R, O, T) EXPAND_OP_4(FUNC, R, T, __VA_ARGS__)
#define EXPAND_OP_6(FUNC, R, T, O, ...)     EXPAND_TYPE(PP_NARG T, FUNC, R, O, T) EXPAND_OP_5(FUNC, R, T, __VA_ARGS__)
#define EXPAND_OP_7(FUNC, R, T, O, ...)     EXPAND_TYPE(PP_NARG T, FUNC, R, O, T) EXPAND_OP_6(FUNC, R, T, __VA_ARGS__)
#define EXPAND_OP_8(FUNC, R, T, O, ...)     EXPAND_TYPE(PP_NARG T, FUNC, R, O, T) EXPAND_OP_7(FUNC, R, T, __VA_ARGS__)
#define EXPAND_OP_9(FUNC, R, T, O, ...)     EXPAND_TYPE(PP_NARG T, FUNC, R, O, T) EXPAND_OP_8(FUNC, R, T, __VA_ARGS__)
#define EXPAND_OP_10(FUNC, R, T, O, ...)    EXPAND_TYPE(PP_NARG T, FUNC, R, O, T) EXPAND_OP_9(FUNC, R, T, __VA_ARGS__)
#define EXPAND_OP_11(FUNC, R, T, O, ...)    EXPAND_TYPE(PP_NARG T, FUNC, R, O, T) EXPAND_OP_10(FUNC, R, T, __VA_ARGS__)
#define EXPAND_OP_12(FUNC, R, T, O, ...)    EXPAND_TYPE(PP_NARG T, FUNC, R, O, T) EXPAND_OP_11(FUNC, R, T, __VA_ARGS__)
#define EXPAND_OP_13(FUNC, R, T, O, ...)    EXPAND_TYPE(PP_NARG T, FUNC, R, O, T) EXPAND_OP_12(FUNC, R, T, __VA_ARGS__)
#define EXPAND_OP_14(FUNC, R, T, O, ...)    EXPAND_TYPE(PP_NARG T, FUNC, R, O, T) EXPAND_OP_13(FUNC, R, T, __VA_ARGS__)
#define EXPAND_OP_15(FUNC, R, T, O, ...)    EXPAND_TYPE(PP_NARG T, FUNC, R, O, T) EXPAND_OP_14(FUNC, R, T, __VA_ARGS__)
#define EXPAND_OP_16(FUNC, R, T, O, ...)    EXPAND_TYPE(PP_NARG T, FUNC, R, O, T) EXPAND_OP_15(FUNC, R, T, __VA_ARGS__)
#define EXPAND_OP_17(FUNC, R, T, O, ...)    EXPAND_TYPE(PP_NARG T, FUNC, R, O, T) EXPAND_OP_16(FUNC, R, T, __VA_ARGS__)
#define EXPAND_OP_18(FUNC, R, T, O, ...)    EXPAND_TYPE(PP_NARG T, FUNC, R, O, T) EXPAND_OP_17(FUNC, R, T, __VA_ARGS__)
#define EXPAND_OP_19(FUNC, R, T, O, ...)    EXPAND_TYPE(PP_NARG T, FUNC, R, O, T) EXPAND_OP_18(FUNC, R, T, __VA_ARGS__)

#define EXPAND_OP_(NARGS, ...)              EXPAND_OP_ ## NARGS (__VA_ARGS__)
#define EXPAND_OP(NARGS, FUNC, R, O, T)     EXPAND_OP_(NARGS, FUNC, R, T, ARG_LIST O)

// Expand the function extension, for example, rank (vector vs scalar)
#define EXPAND_EXT_1(FUNC, O, T, R, ...)    EXPAND_OP(PP_NARG O, FUNC, R, O, T)
#define EXPAND_EXT_2(FUNC, O, T, R, ...)    EXPAND_OP(PP_NARG O, FUNC, R, O, T) EXPAND_EXT_1(FUNC, O, T, __VA_ARGS__)
#define EXPAND_EXT_3(FUNC, O, T, R, ...)    EXPAND_OP(PP_NARG O, FUNC, R, O, T) EXPAND_EXT_2(FUNC, O, T, __VA_ARGS__)
#define EXPAND_EXT_4(FUNC, O, T, R, ...)    EXPAND_OP(PP_NARG O, FUNC, R, O, T) EXPAND_EXT_3(FUNC, O, T, __VA_ARGS__)
#define EXPAND_EXT_5(FUNC, O, T, R, ...)    EXPAND_OP(PP_NARG O, FUNC, R, O, T) EXPAND_EXT_4(FUNC, O, T, __VA_ARGS__)
#define EXPAND_EXT_6(FUNC, O, T, R, ...)    EXPAND_OP(PP_NARG O, FUNC, R, O, T) EXPAND_EXT_5(FUNC, O, T, __VA_ARGS__)
#define EXPAND_EXT_7(FUNC, O, T, R, ...)    EXPAND_OP(PP_NARG O, FUNC, R, O, T) EXPAND_EXT_6(FUNC, O, T, __VA_ARGS__)
#define EXPAND_EXT_8(FUNC, O, T, R, ...)    EXPAND_OP(PP_NARG O, FUNC, R, O, T) EXPAND_EXT_7(FUNC, O, T, __VA_ARGS__)
#define EXPAND_EXT_9(FUNC, O, T, R, ...)    EXPAND_OP(PP_NARG O, FUNC, R, O, T) EXPAND_EXT_8(FUNC, O, T, __VA_ARGS__)
#define EXPAND_EXT_10(FUNC, O, T, R, ...)   EXPAND_OP(PP_NARG O, FUNC, R, O, T) EXPAND_EXT_9(FUNC, O, T, __VA_ARGS__)
#define EXPAND_EXT_11(FUNC, O, T, R, ...)   EXPAND_OP(PP_NARG O, FUNC, R, O, T) EXPAND_EXT_10(FUNC, O, T, __VA_ARGS__)
#define EXPAND_EXT_12(FUNC, O, T, R, ...)   EXPAND_OP(PP_NARG O, FUNC, R, O, T) EXPAND_EXT_11(FUNC, O, T, __VA_ARGS__)
#define EXPAND_EXT_13(FUNC, O, T, R, ...)   EXPAND_OP(PP_NARG O, FUNC, R, O, T) EXPAND_EXT_12(FUNC, O, T, __VA_ARGS__)
#define EXPAND_EXT_14(FUNC, O, T, R, ...)   EXPAND_OP(PP_NARG O, FUNC, R, O, T) EXPAND_EXT_13(FUNC, O, T, __VA_ARGS__)
#define EXPAND_EXT_15(FUNC, O, T, R, ...)   EXPAND_OP(PP_NARG O, FUNC, R, O, T) EXPAND_EXT_14(FUNC, O, T, __VA_ARGS__)
#define EXPAND_EXT_16(FUNC, O, T, R, ...)   EXPAND_OP(PP_NARG O, FUNC, R, O, T) EXPAND_EXT_15(FUNC, O, T, __VA_ARGS__)
#define EXPAND_EXT_17(FUNC, O, T, R, ...)   EXPAND_OP(PP_NARG O, FUNC, R, O, T) EXPAND_EXT_16(FUNC, O, T, __VA_ARGS__)
#define EXPAND_EXT_18(FUNC, O, T, R, ...)   EXPAND_OP(PP_NARG O, FUNC, R, O, T) EXPAND_EXT_17(FUNC, O, T, __VA_ARGS__)
#define EXPAND_EXT_19(FUNC, O, T, R, ...)   EXPAND_OP(PP_NARG O, FUNC, R, O, T) EXPAND_EXT_18(FUNC, O, T, __VA_ARGS__)

#define EXPAND_EXT_(NARGS, ...)             EXPAND_EXT_ ## NARGS (__VA_ARGS__)
#define EXPAND_EXT(NARGS, FUNC, R, O, T)    EXPAND_EXT_(NARGS, FUNC, O, T, ARG_LIST R)

// Instantiate the functions iteratively over three categories that are passed to the provided
// function template. The macros use ext, op, and type for these categories, but they are freely
// definable.
//
// The arguments could be, for example, the "rank" (scalar vs vector), operator (Add, Sub), and
// argument type (float, int, etc.). The values to iterate over must be provided in parenthesis,
// for example:
//
//   INSTANTIATE_FUNCTIONS(binary-function-template, (SV, VS), (Add, Sub), (float, int))
//
#define INSTANTIATE_FUNCTIONS(FUNC, R, O, T)  EXPAND_EXT(PP_NARG R, FUNC, R, O, T)

#endif  // TENSOR_METAL_KERNELS_UTILS_H
