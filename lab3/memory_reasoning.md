Questions from lab: Walk through the following calculations in the context of the “large-scale tests” for the naive GPU implementation, which evaluate it on a domain of size 1601 * 1601 for 6400 timesteps:

Q1: How does the total size of the u0 and u1 buffers compare to the capacity of the L2 cache?

A1: Each u0 and u1 buffer is of size 1601 * 1601 * 4 = 10.25MB, so both buffers combined are approx. 20.50MB. The L2 cache is only 4MB, so both of our buffers are much larger than the L2 cache.

Q2: Assuming the kernel makes no use of the L1 cache, roughly how many bytes is each launch of wave_gpu_naive_step requesting to load / store in the L2 cache?

A2: We read from u1 5 times (at idx, and once from up, down, left, and right of idx) and from u0 once at idx. We write to u0 as well, just one time. This is 7 floats to load/store (7 * 4 = 28 bytes), for each launch for 1601^2. Thus, we have 1601^2 * 28 bytes =~= 72MB. 

Q3: Of those requests to L2, roughly how many bytes’ worth of requests miss L2 and get through to DRAM?

A3: I'm under the impression that most of these 72GB of requests would miss L2. We have many threads accessing this data in parallel; this places the cache under pressure as cache lines would constantly be filled and evicted. 

Q4: Given the number of bytes’ worth of requests that get through to DRAM, roughly how long would the naive GPU simulation take to run if the only constraint were DRAM bandwidth?

A4: DRAM moves memory at 360GB/sec, with 6400 timesteps, we have 72MB * (10^{-3} GB/MB) / (360GB/s) * 6400 = 1.280s.

Q5: Similarly, roughly how long would it take to run if the only constraint were L2 bandwidth?

A5: Using the same math, 72MB * (10^{-6} MB/TB) / (2.5TB/s) * 6400 = 0.184s.

Q6: How do those estimates compare to your naive GPU implementation’s actual run time?

Does your answer to (6) have any implications for attempts to optimize the implementation further?

A6: 