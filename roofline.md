Part 0: Prelab – Roofline

In previous prelabs, we have benchmarked the latency of both arithmetic and memory instructions to determine the relative latency of operations, which provide an insight into the internal operations of the hardware we are using. In this prelab, we’ll take a higher level view of the same idea by diving deeper into the Roofline Model, which provides a simple model for performance engineering.

Deliverable: Read the explanation of the Roofline Model, and answer the two prelab questions below.

Prelab Question 1: What is the operational intensity of a matrix multiplication where M=N=K=n? (the equation may depend on n)? You may want to consider multiple cases of how the values in the matrices are loaded from memory during the computation.

Q1 Answer: For matrices of this dimension, we do n multiplications and n additions (in the pattern below where we start sum=0), so we have approximately 2n FLOPs per row of the left matrix (w.l.o.g.), then EACH row of the left matrix is multiplied by EACH column of the right matrix, so we repeat this process n^2 times, leaving us with 2n^3 FLOPs per multiplication.

Meanwhile, in an optimal case (in which we read every byte only once) we read n^2 * d bytes for the matrices we are multiplying (where d is the size of whatever data type we are dealing with) and we write n^2 * d bytes for our output, so we have 3n^2 * d bytes in total written. 

Thus, (optimally) we end up with an operational intensity of 2/(3*d) * n. 

To envision a worse scenario, we can imagine that for each element in the output matrix, we load O(n) values (e.g. we store nothing and load the row of A and column of B we are dealing with). Thus, for each of the n^2 output values, we are loading O(n) values so we end up with O(n^3) bytes accessed, leaving us with a CONSTANT operational intensity. 

Prelab Question 2: For what value of n does matrix multiplication transition from being memory-bound to compute-bound on our GPU? You may want to consider multiple cases of how the values in the matrices are loaded from memory during the computation.

Q2 Answer: For starters, imagine our OI is constant like in the bad scenario; in this case, we can imagine that we will NEVER be compute bound as we do OI * bandwidth bytes per second; this multiplied by the GPU bandwidth will conceivably never reach the peak throughput, which is measured in TFlops/sec. 

Instead presume we are dealing with floats (4 bytes) and the optimal case, where our OI is n/6. Here, there are a variety of access patterns we can consider. 

If we were to load everything from DRAM, we would solve for n s.t. n/6 FLOPs/byte * 360 GB/s = 0.06nTFLOP/s = 26.7 TFLOP/s --> n=445.

If we instead load from L2, we have n/6 FLOPs/byte * 2.4 TB/s = 26.7 TFLOPs --> n = 67.

-------------------------------

Question 1 for final write-up: Walk through the following analysis in the context of the large-scale tests, with matrices of size 3072:

1) Using your calculation of operational intensity from the prelab, what is the fastest this workload could possibly run on our GPU if it was compute-bound? (Assume we are only using FMAs and no Tensor Cores.)

Answer 1: Because we are compute-bound, we know we are processing 26.7 TFLOP/sec. All that remains is to count the number of TFLOPs; assuming floats, we know we have 2*(3072)^3 = 0.0580 TFLOPs, so this would take 2.17ms.

2) Similarly, assuming we only need to access each unique location in DRAM once, what is the fastest this workload could possibly run if it was DRAM-bound?

Answer 2: We know we are compute bound here too; we instead can calculate the amount of time it takes to move all the bytes. 3n^2 * 4=3(3072)^2 = 0.028GB are written, so at 0.028GB/360GB/s we get a time of 0.31ms.

3) How does (2) compare to (1)? Given a very well-optimized implementation, would you expect the run time of this workload to be dominated by compute or by data movement?

Answer 3: (2) is much faster than (1) by an order of magnitude, so if this were compute-free, we would have a lower bound of 0.31ms from DRAM. 

4) Alternatively, imagine we do not exploit reuse, so every s += a[i, k] * b[k, j] operation loads the a[i, k] and b[k, j] elements directly from DRAM. Then how many total bytes would we need to load from DRAM?

Answer 4: For each element, we have 2n * 4 bytes loaded + 4 bytes written; this ends up being over n^2 entries so we have (8n + 4) * n^2 = 232 GB.

5) If we had no reuse, as in (4), and if DRAM bandwidth were the only constraint, what is the fastest this workload could possibly run on our GPU?

Answer 5: We have 232GB/360GB/s=0.64s. Way slower.

6) Imagine instead that every s += a[i, k] * b[k, j] operation could somehow load directly from L2. Then if L2 bandwidth were the only constraint, what is the fastest this workload could possibly run?

Answer 6: Assuming no reuse. 0.232TB bytes, loaded at a rate 2.4TB/s, leaves us with 0.232TB/2.4TB/s=0.096s=96ms. Faster!

7) How do (5) and (6) compare to (1)?

Answer 7: These times are much longer than the idal limit in 1. We have a pretty good theoretical bound now.

