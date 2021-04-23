# GPU Simulation Assignment Part #3

## Implementing GPU prefetching


In this part of the assignment, you will implement a simple GPU prefetching technique in the
following MICRO paper:

```
J. Lee, H. Kim, N. B. Lakshminarayana and R. Vuduc,
"Many-Thread Aware Prefetching Mechanisms for GPGPU Applications,"
in IEEE/ACM International Symposium on Microarchitecture (MICRO), 2010.
```

This paper describes the tradeoffs in enabling data prefetching for GPUs.
For this part of the assignment - use PTX mode, otherwise you will
have to generate traces from real hardware to operate in SASS-mode
on the microbenchmark you are asked to create in the next part.

### Implementing a simple microbenchmark (uBench)

To verify your changes in the rest of this assignment work as intended,
write a CUDA microbenchmark that has an easy to detect memory access pattern that is amenable to prefetching.
Measure the IPC and L1/L2 misses using the V100 config, with memory copying into the L2 turned off, making sure that:

```
# This 1 by default. When on, all cudamemcopies will go into the L2 cache, making prefetching less effective.
# For the rest of this assignment, turn this flag off.
-gpgpu_perf_sim_memcpy 0
```

You can use all the same infrastructure for launching jobs by adding your uBench to:
```
util/job_launching/apps/define-all-apps.yml
```

add something like:

```
# Assuming your executable is named uBench
# and there are no arguments
uBench:
    exec_dir: "UBENCH_EXE_LOCATION"
    data_dirs: ""
    execs:
        - uBench:
            - args:
```

### Implementing the simple prefetcher

Modify the simulator to implement a many thread aware prefetch mechanism like that described in the paper.
To keep the assignment simple, you are permitted to simplify the training mechanism so that it works
only with your microbenchmark.

Measure the performance change when your prefetch algorithm is enabled versus when it is disabled.
Plot the relative IPC, cache misses and absolute number of prefetch hits (you need to implement a counter for this).

## What to hand in

Submit a 2-page maximum pdf that includes the data plots requested.
Be sure to include a paragraph the describes the data and the reasons why it looks
the way it does.

