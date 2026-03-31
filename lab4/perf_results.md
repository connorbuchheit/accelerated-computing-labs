matmul_l1:

  size  256 *  256 *  256:
    correctness: 8.26e-08 relative RMSE
    run time:   0.01 ms
    throughput:  2.72 TFLOP/s

  size 3072 * 3072 * 3072:
    correctness: 1.03e-06 relative RMSE
    run time:  12.85 ms
    throughput:  4.51 TFLOP/s

matmul_l1_reg:

  size  256 *  256 *  256:
    correctness: 8.26e-08 relative RMSE
    run time:   0.05 ms
    throughput:  0.73 TFLOP/s

  size 3072 * 3072 * 3072:
    correctness: 1.03e-06 relative RMSE
    run time:   3.00 ms
    throughput: 19.33 TFLOP/s

speedups on largest problem size:

  speedup matmul_l1 -> matmul_l1_reg: 4.28x