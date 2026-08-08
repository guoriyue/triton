// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' -verify-diagnostics

#blocked0 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#mma0 = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx = 0, parent = #mma0, kWidth = 1}>
#dot_operand_b = #ttg.dot_op<{opIdx = 1, parent = #mma0, kWidth = 1}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  tt.func public @test_mmav2_dot_unsupported_type(%A: tensor<16x4xf64, #blocked0>, %B: tensor<4x16xf64, #blocked0>) -> tensor<16x16xi32, #mma0> {
    %AA = ttg.local_alloc %A : (tensor<16x4xf64, #blocked0>) -> !ttg.memdesc<16x4xf64, #shared0, #smem>
    %BB = ttg.local_alloc %B : (tensor<4x16xf64, #blocked0>) -> !ttg.memdesc<4x16xf64, #shared0, #smem>
    %AA_DOT = ttg.local_load %AA : !ttg.memdesc<16x4xf64, #shared0, #smem> -> tensor<16x4xf64, #dot_operand_a>
    %BB_DOT = ttg.local_load %BB : !ttg.memdesc<4x16xf64, #shared0, #smem> -> tensor<4x16xf64, #dot_operand_b>
    %cst0 = arith.constant dense<0> : tensor<16x16xi32, #mma0>
    // expected-error@+2 {{unsupported MMA instruction for the given operand/result types}}
    // expected-error@+1 {{failed to legalize operation 'tt.dot' that was explicitly marked illegal}}
    %D = tt.dot %AA_DOT, %BB_DOT, %cst0 : tensor<16x4xf64, #dot_operand_a> * tensor<4x16xf64, #dot_operand_b> -> tensor<16x16xi32, #mma0>
    tt.return %D : tensor<16x16xi32, #mma0>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CGALayout = [[1, 0]], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[1, 0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_wgmma_multi_cta_smem_operand(
      %a: tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>,
      %b: !ttg.memdesc<128x128xbf16, #shared, #smem>) {
    %acc = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    // The non-zero CTA block basis used to abort in the SMEM descriptor loader.
    // expected-error@+2 {{MMA operand shared-memory load does not support a non-zero CTA block offset (num_ctas > 1)}}
    // expected-error@+1 {{failed to legalize operation 'ttng.warp_group_dot' that was explicitly marked illegal}}
    %res = ttng.warp_group_dot %a, %b, %acc :
      tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> *
      !ttg.memdesc<128x128xbf16, #shared, #smem> -> tensor<128x128xf32, #mma>
    tt.return
  }
}
