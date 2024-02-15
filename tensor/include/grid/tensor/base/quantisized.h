


namespace matmul {

// q4 -> q8

template <>
inline q8_t VecDot<q8_t, q4_t>(const q4_t* src1, const q4_t* src2, size_t dimensions) const
{
  
}

// FIXME: dequantisize through copy? Copy<float, q4_t>, etc.
// FIXME: one option could also be:  result = tensor.ViewDouble() op ... op ...
//        the op takes the doub
// FIXME: need to specify both types? maybe because of promotion?
template <DEV>
requires <DEV::has_xyz>
{

}

// ...


} // end of namespace matmul

