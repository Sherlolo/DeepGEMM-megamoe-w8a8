# W4A8 Source Provenance

The W4A8 GEMM and intrinsics are adapted from the user-specified local DeepGEMM repository at commit edf18625ad609e2be8f9b8a91076e422ec223204:

- csrc/arch/gfx936_gfx938_shared/m_grouped_w4a8_gemm_nt_contiguous_hipc.cu
- csrc/include/hipc_w4a8/w4a8_intrinsics.h
- csrc/include/hipc_w4a8/w4a8_nwave_bm128_bn512.h (direct vector A-to-LDS instruction, descriptor stride, and matching LDS read layout)
- csrc/include/moe_marlin/intrinsic.h (direct-to-LDS buffer load instruction form)

Initial adaptation removes unused includes and isolates the host entry point in namespace megamoe_w4a8. Arithmetic and packed weight layout are preserved. No source LICENSE file or license header exists in these two source files; redistribution terms need to be obtained from that repository owner before external publication. This task only modifies the user's local checkout.
