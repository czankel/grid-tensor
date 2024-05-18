//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef TENSOR_SOURCE_INSTANTIATE_H
#define TENSOR_SOURCE_INSTANTIATE_H

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

// Instantiate the actual function for the number of iteration groups

#define DEFINE_1(FN, X, Y, Z) FN(Z)
#define DEFINE_2(FN, X, Y, Z) FN(Y, Z)
#define DEFINE_3(FN, X, Y, Z) FN(X, Y, Z)

// Expand one argument group

#define ITER1_1(FN, CARGS, X, Y, Z, ...)  DEFINE_##CARGS(FN, X, Y, Z)
#define ITER1_2(FN, CARGS, X, Y, Z, ...)  DEFINE_##CARGS(FN, X, Y, Z) ITER1_1(FN, CARGS, X, Y, __VA_ARGS__)
#define ITER1_3(FN, CARGS, X, Y, Z, ...)  DEFINE_##CARGS(FN, X, Y, Z) ITER1_2(FN, CARGS, X, Y, __VA_ARGS__)
#define ITER1_4(FN, CARGS, X, Y, Z, ...)  DEFINE_##CARGS(FN, X, Y, Z) ITER1_3(FN, CARGS, X, Y, __VA_ARGS__)
#define ITER1_5(FN, CARGS, X, Y, Z, ...)  DEFINE_##CARGS(FN, X, Y, Z) ITER1_4(FN, CARGS, X, Y, __VA_ARGS__)
#define ITER1_6(FN, CARGS, X, Y, Z, ...)  DEFINE_##CARGS(FN, X, Y, Z) ITER1_5(FN, CARGS, X, Y, __VA_ARGS__)
#define ITER1_7(FN, CARGS, X, Y, Z, ...)  DEFINE_##CARGS(FN, X, Y, Z) ITER1_6(FN, CARGS, X, Y, __VA_ARGS__)
#define ITER1_8(FN, CARGS, X, Y, Z, ...)  DEFINE_##CARGS(FN, X, Y, Z) ITER1_7(FN, CARGS, X, Y, __VA_ARGS__)
#define ITER1_9(FN, CARGS, X, Y, Z, ...)  DEFINE_##CARGS(FN, X, Y, Z) ITER1_8(FN, CARGS, X, Y, __VA_ARGS__)
#define ITER1_10(FN, CARGS, X, Y, Z, ...) DEFINE_##CARGS(FN, X, Y, Z) ITER1_9(FN, CARGS, X, Y, __VA_ARGS__)
#define ITER1_11(FN, CARGS, X, Y, Z, ...) DEFINE_##CARGS(FN, X, Y, Z) ITER1_10(FN, CARGS, X, Y, __VA_ARGS__)
#define ITER1_12(FN, CARGS, X, Y, Z, ...) DEFINE_##CARGS(FN, X, Y, Z) ITER1_11(FN, CARGS, X, Y, __VA_ARGS__)
#define ITER1_13(FN, CARGS, X, Y, Z, ...) DEFINE_##CARGS(FN, X, Y, Z) ITER1_12(FN, CARGS, X, Y, __VA_ARGS__)
#define ITER1_14(FN, CARGS, X, Y, Z, ...) DEFINE_##CARGS(FN, X, Y, Z) ITER1_13(FN, CARGS, X, Y, __VA_ARGS__)
#define ITER1_15(FN, CARGS, X, Y, Z, ...) DEFINE_##CARGS(FN, X, Y, Z) ITER1_14(FN, CARGS, X, Y, __VA_ARGS__)
#define ITER1_16(FN, CARGS, X, Y, Z, ...) DEFINE_##CARGS(FN, X, Y, Z) ITER1_15(FN, CARGS, X, Y, __VA_ARGS__)
#define ITER1_17(FN, CARGS, X, Y, Z, ...) DEFINE_##CARGS(FN, X, Y, Z) ITER1_16(FN, CARGS, X, Y, __VA_ARGS__)
#define ITER1_18(FN, CARGS, X, Y, Z, ...) DEFINE_##CARGS(FN, X, Y, Z) ITER1_17(FN, CARGS, X, Y, __VA_ARGS__)
#define ITER1_19(FN, CARGS, X, Y, Z, ...) DEFINE_##CARGS(FN, X, Y, Z) ITER1_18(FN, CARGS, X, Y, __VA_ARGS__)

#define ITER1_(I, ...)                    ITER1_##I(__VA_ARGS__)
#define ITER1(I, FN, CARGS, X, Y, Z)      ITER1_(I, FN, CARGS, X, Y, ARG_LIST Z)

// Expand two argument groups
#define ITER2_1(FN, CARGS, X, Z, Y, ...)  ITER1(PP_NARG Z, FN, CARGS, X, Y, Z)
#define ITER2_2(FN, CARGS, X, Z, Y, ...)  ITER1(PP_NARG Z, FN, CARGS, X, Y, Z) ITER2_1(FN, CARGS, X, Z, __VA_ARGS__)
#define ITER2_3(FN, CARGS, X, Z, Y, ...)  ITER1(PP_NARG Z, FN, CARGS, X, Y, Z) ITER2_2(FN, CARGS, X, Z, __VA_ARGS__)
#define ITER2_4(FN, CARGS, X, Z, Y, ...)  ITER1(PP_NARG Z, FN, CARGS, X, Y, Z) ITER2_3(FN, CARGS, X, Z, __VA_ARGS__)
#define ITER2_5(FN, CARGS, X, Z, Y, ...)  ITER1(PP_NARG Z, FN, CARGS, X, Y, Z) ITER2_4(FN, CARGS, X, Z, __VA_ARGS__)
#define ITER2_6(FN, CARGS, X, Z, Y, ...)  ITER1(PP_NARG Z, FN, CARGS, X, Y, Z) ITER2_5(FN, CARGS, X, Z, __VA_ARGS__)
#define ITER2_7(FN, CARGS, X, Z, Y, ...)  ITER1(PP_NARG Z, FN, CARGS, X, Y, Z) ITER2_6(FN, CARGS, X, Z, __VA_ARGS__)
#define ITER2_8(FN, CARGS, X, Z, Y, ...)  ITER1(PP_NARG Z, FN, CARGS, X, Y, Z) ITER2_7(FN, CARGS, X, Z, __VA_ARGS__)
#define ITER2_9(FN, CARGS, X, Z, Y, ...)  ITER1(PP_NARG Z, FN, CARGS, X, Y, Z) ITER2_8(FN, CARGS, X, Z, __VA_ARGS__)
#define ITER2_10(FN, CARGS, X, Z, Y, ...) ITER1(PP_NARG Z, FN, CARGS, X, Y, Z) ITER2_9(FN, CARGS, X, Z, __VA_ARGS__)
#define ITER2_11(FN, CARGS, X, Z, Y, ...) ITER1(PP_NARG Z, FN, CARGS, X, Y, Z) ITER2_10(FN, CARGS, X, Z, __VA_ARGS__)
#define ITER2_12(FN, CARGS, X, Z, Y, ...) ITER1(PP_NARG Z, FN, CARGS, X, Y, Z) ITER2_11(FN, CARGS, X, Z, __VA_ARGS__)
#define ITER2_13(FN, CARGS, X, Z, Y, ...) ITER1(PP_NARG Z, FN, CARGS, X, Y, Z) ITER2_12(FN, CARGS, X, Z, __VA_ARGS__)
#define ITER2_14(FN, CARGS, X, Z, Y, ...) ITER1(PP_NARG Z, FN, CARGS, X, Y, Z) ITER2_13(FN, CARGS, X, Z, __VA_ARGS__)
#define ITER2_15(FN, CARGS, X, Z, Y, ...) ITER1(PP_NARG Z, FN, CARGS, X, Y, Z) ITER2_14(FN, CARGS, X, Z, __VA_ARGS__)
#define ITER2_16(FN, CARGS, X, Z, Y, ...) ITER1(PP_NARG Z, FN, CARGS, X, Y, Z) ITER2_15(FN, CARGS, X, Z, __VA_ARGS__)
#define ITER2_17(FN, CARGS, X, Z, Y, ...) ITER1(PP_NARG Z, FN, CARGS, X, Y, Z) ITER2_16(FN, CARGS, X, Z, __VA_ARGS__)
#define ITER2_18(FN, CARGS, X, Z, Y, ...) ITER1(PP_NARG Z, FN, CARGS, X, Y, Z) ITER2_17(FN, CARGS, X, Z, __VA_ARGS__)
#define ITER2_19(FN, CARGS, X, Z, Y, ...) ITER1(PP_NARG Z, FN, CARGS, X, Y, Z) ITER2_18(FN, CARGS, X, Z, __VA_ARGS__)

#define ITER2_(I, ...)                    ITER2_##I(__VA_ARGS__)
#define ITER2(I, FN, CARGS, X, Y, Z)      ITER2_(I, FN, CARGS, X, Z, ARG_LIST Y)

// Expand three argument groups
#define ITER3_1(FN, CARGS, Y, Z, X, ...)  ITER2(PP_NARG Y, FN, CARGS, X, Y, Z)
#define ITER3_2(FN, CARGS, Y, Z, X, ...)  ITER2(PP_NARG Y, FN, CARGS, X, Y, Z) ITER3_1(FN, CARGS, Y, Z, __VA_ARGS__)
#define ITER3_3(FN, CARGS, Y, Z, X, ...)  ITER2(PP_NARG Y, FN, CARGS, X, Y, Z) ITER3_2(FN, CARGS, Y, Z, __VA_ARGS__)
#define ITER3_4(FN, CARGS, Y, Z, X, ...)  ITER2(PP_NARG Y, FN, CARGS, X, Y, Z) ITER3_3(FN, CARGS, Y, Z, __VA_ARGS__)
#define ITER3_5(FN, CARGS, Y, Z, X, ...)  ITER2(PP_NARG Y, FN, CARGS, X, Y, Z) ITER3_4(FN, CARGS, Y, Z, __VA_ARGS__)
#define ITER3_6(FN, CARGS, Y, Z, X, ...)  ITER2(PP_NARG Y, FN, CARGS, X, Y, Z) ITER3_5(FN, CARGS, Y, Z, __VA_ARGS__)
#define ITER3_7(FN, CARGS, Y, Z, X, ...)  ITER2(PP_NARG Y, FN, CARGS, X, Y, Z) ITER3_6(FN, CARGS, Y, Z, __VA_ARGS__)
#define ITER3_8(FN, CARGS, Y, Z, X, ...)  ITER2(PP_NARG Y, FN, CARGS, X, Y, Z) ITER3_7(FN, CARGS, Y, Z, __VA_ARGS__)
#define ITER3_9(FN, CARGS, Y, Z, X, ...)  ITER2(PP_NARG Y, FN, CARGS, X, Y, Z) ITER3_8(FN, CARGS, Y, Z, __VA_ARGS__)
#define ITER3_10(FN, CARGS, Y, Z, X, ...) ITER2(PP_NARG Y, FN, CARGS, X, Y, Z) ITER3_9(FN, CARGS, Y, Z, __VA_ARGS__)
#define ITER3_11(FN, CARGS, Y, Z, X, ...) ITER2(PP_NARG Y, FN, CARGS, X, Y, Z) ITER3_10(FN, CARGS, Y, Z, __VA_ARGS__)
#define ITER3_12(FN, CARGS, Y, Z, X, ...) ITER2(PP_NARG Y, FN, CARGS, X, Y, Z) ITER3_11(FN, CARGS, Y, Z, __VA_ARGS__)
#define ITER3_13(FN, CARGS, Y, Z, X, ...) ITER2(PP_NARG Y, FN, CARGS, X, Y, Z) ITER3_12(FN, CARGS, Y, Z, __VA_ARGS__)
#define ITER3_14(FN, CARGS, Y, Z, X, ...) ITER2(PP_NARG Y, FN, CARGS, X, Y, Z) ITER3_13(FN, CARGS, Y, Z, __VA_ARGS__)
#define ITER3_15(FN, CARGS, Y, Z, X, ...) ITER2(PP_NARG Y, FN, CARGS, X, Y, Z) ITER3_14(FN, CARGS, Y, Z, __VA_ARGS__)
#define ITER3_16(FN, CARGS, Y, Z, X, ...) ITER2(PP_NARG Y, FN, CARGS, X, Y, Z) ITER3_15(FN, CARGS, Y, Z, __VA_ARGS__)
#define ITER3_17(FN, CARGS, Y, Z, X, ...) ITER2(PP_NARG Y, FN, CARGS, X, Y, Z) ITER3_16(FN, CARGS, Y, Z, __VA_ARGS__)
#define ITER3_18(FN, CARGS, Y, Z, X, ...) ITER2(PP_NARG Y, FN, CARGS, X, Y, Z) ITER3_17(FN, CARGS, Y, Z, __VA_ARGS__)
#define ITER3_19(FN, CARGS, Y, Z, X, ...) ITER2(PP_NARG Y, FN, CARGS, X, Y, Z) ITER3_18(FN, CARGS, Y, Z, __VA_ARGS__)

#define ITER3_(I, ...)                    ITER3_##I(__VA_ARGS__)
#define ITER3(I, FN, CARGS, X, Y, Z)      ITER3_(I, FN, CARGS, Y, Z, ARG_LIST X)

// INSTANTIATE<X> instantiate the provided functions with up to three iteration groups.
//
// The iteration arguments must be passed in braces, for example: "( Arg1, Arg2 )"
//
// The arguments could be, for example, the "rank" (scalar vs vector), operator (Add, Sub), and
// argument type (float, int, etc.). The values to iterate over must be provided in parenthesis,
// for example:
//
//   INSTANTIATE_FUNCTIONS(BinaryFunctionTemplate, (ScalarVector, VectorScalar), (Add, Sub), (float, int))
//
// This instantiates 6 functions (2 * 2 * 2 arguments):
//
//   BinaryFunctionTemplate(ScalarVector, Add, float);
//   BinaryFunctionTemplate(ScalarVector, Add, int);
//   ...
//   BinaryFunctionTemplate(VectorScalar, Sub, int);
//
//

#define INSTANTIATE3(FN, X, Y, Z) ITER3(PP_NARG X, FN, 3, X, Y, Z)
#define INSTANTIATE2(FN, X, Y)    ITER2(PP_NARG X, FN, 2,  , X, Y)
#define INSTANTIATE1(FN, X)       ITER1(PP_NARG X, FN, 1,  ,  , X)

#endif  // TENSOR_SOURCE_INSTANTIATE_H
