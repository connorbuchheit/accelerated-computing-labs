Questions from lab: Walk through the following calculations in the context of the “large-scale tests” for the naive GPU implementation, which evaluate it on a domain of size 1601 * 1601 for 6400 timesteps:

How does the total size of the u0 and u1 buffers compare to the capacity of the L2 cache?

Assuming the kernel makes no use of the L1 cache, roughly how many bytes is each launch of wave_gpu_naive_step requesting to load / store in the L2 cache?

Of those requests to L2, roughly how many bytes’ worth of requests miss L2 and get through to DRAM?

Given the number of bytes’ worth of requests that get through to DRAM, roughly how long would the naive GPU simulation take to run if the only constraint were DRAM bandwidth?

Similarly, roughly how long would it take to run if the only constraint were L2 bandwidth?

How do those estimates compare to your naive GPU implementation’s actual run time?

Does your answer to (6) have any implications for attempts to optimize the implementation further?