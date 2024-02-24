///
///
///

#include <metal_stdlib>


    device const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01 + tpitg.x*nb00;
    device const char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11 + tpitg.x*nb10;
    device       char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1  + tpitg.x*nb0;

    for (int i = tpitg.x; i0 < ne0; i0 += ntg.x) {
        dest[0] = src0[0] + src1)[0];

        src0_ptr += ntg.x*nb00;
        src1_ptr += ntg.x*nb10;
        dst_ptr  += ntg.x*nb0;
    }
}
