# GPU Simulation Assignment Part #1

## Introduction to GPU Simulation

In this assignment we will introduce you to a GPU Simulators: Accel-Sim (https://accel-sim.github.io) and GPGPU-Sim.
Accel-Sim is a trace-based simulation framework that uses another simulator GPGPU-Sim (http://gpgpu-sim.org) as it's core performance model.
Documentation on how to use Accel-Sim can be found in the readme's associated with with the github repo - however we will cover the main points you need here.
There are a variety of documentation sources for GPGPU-Sim, including a somewhat dated manual (http://gpgpu-sim.org/manual).
There are also some pre-recorded tutorial videos here. Note that these videos were made last year and are specific to GPGPU-Sim.
* https://www.youtube.com/watch?v=d2bzYap_fzc
* https://www.youtube.com/watch?v=6v0qSPJH-VE&t=3s

The docs are a good place to get started, but, as with any piece of open-source software - looking at the code is your best bet.
**Never trust a counter you have not traced back in the code to see that it does something sane**.

The goal of a simulator implemented in a high-level language (this one is written in C++),
is to evaluate architectural modifications quickly to determine if
it's worthwhile to invest further time into creating hardware for them.
To that end, the simulator models (as closely as possible) all the relevant microarchitectural actions that happen every cycle of the GPU.
For this portion of the assignment, you will familiarize yourself with the codebase, run some simulations, collect some basic data and do some analysis.

Some notes on the differences between GPGPU-Sim and Accel-Sim.
One major difference is that Accel-Sim uses the real machine ISA instructions (called SASS in NVIDIA machines),
while GPGPU-Sim uses PTX instructions, which are a virtual ISA, or an intermediate representation of the program.
A side-effect of this is that Accel-Sim tends to more accurately model the real program behaviour, as some compiler optimizations
the ISA is 1:1 with hardware and the SASS has extra compilation optimizations not seen in the PTX.
Another major difference between the two simulators is their mode of simulation. Accel-Sim uses instruction traces collected from hardware, whereas
GPGPU-Sim performs functional emulation of the PTX instructions.
Practically this means the Accel-Sim relies of large trace files, while PTX simulation requires all the corresponding application data
and needs the application binary compiled. There are advantages and disadvantages to both approaches. For this assignment,
we will use the execution-driven GPGPU-Sim simulation.

## Running your first program.

### Setup

There are a few dependencies you need to take care of to run the simulator.
On qstruct, most of the packages are installed, but if you want to run things locally, on Ubuntu the necessary packages are:

```shell
# For a home machine
sudo apt-get install build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev doxygen graphviz python-pmw python-ply python-numpy libpng12-dev python-matplotlib
```

On your private machine, you will need to install CUDA:

```bash
# For a home machine
mkdir cuda && cd cuda
wget https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_387.26_linux
sh cuda_9.1.85_387.26_linux --silent --toolkit --toolkitpath=`pwd` 
export CUDA_INSTALL_PATH=`pwd`
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
```

On qstruct, CUDA is pre-installed in the class account, but you will need to run the following commands to set it up:

```bash
# Required on qstruct
export CUDA_INSTALL_PATH=/home/min/a/ee695paa/cuda
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
```

To run GPGPU-Sim in execution-driven mode, you will need some CUDA apps.
We have provided the apps in the 695 account directory and you can link everything by running:
```bash
# Required on qstruct
source /home/min/a/ee695paa/github/gpu-app-collection/src/setup_environment
```
To build stuff for yourself on a home machine:
```bash
# For a home machine
git clone https://github.com/accel-sim/gpu-app-collection.git
cd gpu-app-collection
source ./src/setup_environment
make all -i -j -C ./src
```

To get the simulator:

Start by cloning the templated repo for this assignment in the same way you have previously.

### Building

To build it the simulator:

```bash
cd gpu-simulator
source setup_environment.sh
make -j
```

For iterative builds you can just use "make" and skip the sourcing.

### Executing the simulator

GPGPU-Sim operates by tricking CUDA executables into using the simulator instead of a real GPU.
The simulator itself is compiled as a "shared object" in linux (libcudart.so) - that mirrors the libcudart.so provided by NVIDIA.
When you source the setup\_environment, we put the path to the libcudart.so in the LD\_LIBRARY\_PATH.
So from then on, in that shell, you can run any CUDA executable and GPGPU-Sim will be used.
The Accel-Sim framework provides some convinient scripts for running the apps and collecting simulation output.

```bash
# To kick off some simulations
# -C specifies which machine to simulate, -B is the benchmark suite (we have a pre-defined set of apps for this assingment)
# and -N allow you to name this run so you can collect stats, and keep track of what is done and what is still running.
# -c limits the number of cores you are going to use. For now - please limit to 10 -- this allow 4 people to use qstruct at the same time.
# As we get closer to the deadline, I will ask folks to dail this number down further.
# the launching system has persistence and all the jobs you launch with run_simulations will be counted.
./util/job_launching/run_simulations.py -C QV100-PTX-1B_INSN -B ece695.part1 -N ptx-run -c 10
```

This will launch processes running the simulator in the background.
To monitor what is done, you can use the following command:

```bash
./util/job_launching/job_status.py -N ptx-run
```


## Objective: Data collection

The objective of part 1 is fairly simple: run the simulator under a few different configuarations, play with
some options, collect the results and analyze them.

### Data Point \# 1

Run the simulation in PTX-driven mode as specified above, then run those same apps in SASS-driven execution.
Collect the total cycles and instructions simulated. For this data point answer the questions:

1. **What do these numbers say about the design of the PTX versus SASS ISA?**
1. **What do these numbers say about the optimization level of each code type?**

#### Hints:

To run in SASS-mode, use the following command:
```bash
./util/job_launching/run_simulations.py -C QV100-SASS-1B_INSN -B ece695.part1 -T /home/min/a/ee695paa/accel-sim-traces/rodinia-3.1/11.2/ -N sass-run -c 10
```

This command will launch the jobs asynchronously. To monitor what jobs are running/complete, you can use:
```bash
./util/job_launching/monitor_func_test.py -N sass-run -v
```

The monitor will run until everything is complete, if you want a 1-shot test of what is done you can just run:
```bash
./util/job_launching/job_status.py -N sass-run
```

To understand a bit of what -C and -B mean, take a look at:
```bash
cat ./util/job_launching/apps/define-695paa-apps.yml
```

and

```bash
less ./util/job_launching/configs/define-standard-cfgs.yml
```

The configs are probably more interesting and relevant for this assignment.
The config string "QV100-SASS-1B\_INSN" is a hyphen-separated list. The first token "QV100" specifies the baseline configuration file
to use. In this case:

```bash
less $GPGPUSIM_ROOT/configs/tested-cfgs/SM7_QV100/gpgpusim.config
```

If you look around this file, all the parameters for the NVIDIA Volta V100 GPU are set.
On top of that baseline config file, there are a variety of other extra tokens, in this simple experiment,
the only one you are using is the "1B\_INSN" flag.
You can add as many extra-params as you like to the config string. In the end, all the configuration parameters are aggregrated
and only the most recent config parameter is set.
There are few tokens pre-specified in this file, but when you are creating your own tests, you can always make your own by
adding to the file.


If you ever want to kill the jobs you launched before they complete, use the following:
```bash
./util/job_launching/procman.py -k
```

All the output from the simulator will be placed in sim\_run\_9.1/(app)/(args)/(config)/ inside each of these directories,
there will be files labeled \*.o(jobId) and \*.e(jobId) which store the stdout (o) and stderr (e) for each particular run.
To help aggregate the data from all the apps, configs, there are some handy scripts that parse all the GPGPU-Sim
output and create CSV files with the final results.

To collect the statistics, you can run:
```bash
./util/job_launching/get_stats.py -C QV100-SASS-1B_INSN,QV100-PTX-1B_INSN -B ece695.part1 -R | tee mystats.csv
```

By default, all the stats collected (which are represented as regexes pulled from the output files)
are listed here, you can always add to this list with other stats you create, etc.

```bash
cat ./util/job_launching/stats/example_stats.yml
```

There are even some convenient plotting scripts for the output of get\_stats, like:

```bash
./util/plotting/plot-get-stats.py -c mystats.csv
```

which will generate a collection of nice looking plots of all your statistics in:

```bash
./util/plotting/htmls/*.html
```

Of course you use the output of get\_stats with any plotting tool you like (excel, matplotlib, whatever)
Or you can write your own stat parsing scripts from the output files, or manually copy/past output from the .o files into
excel. There are no requirements to use these data collection and plotting tools for the assignment, we are
just letting you know they are there and can make your life easier.


### Data Point \# 2

In SASS mode, collect and plot metrics on how much control-flow divergence (average SIMD-efficiency, which ranges between 1/32 and 32/32)
and memory divergence (average number of memory accesses per memory instruction) there is in each application.
For memory, keep in mind that the volta machine has a 128-byte cache lines with 32B sectors. Therefore; a perfectly coalesced
memory instruction can still generate 4 32-byte accesses. In the worst-case, a completely diverged access generates 32 memory 32-byte transactions.

Answer the following questions:

1. **Is there a connection between these divergence metrics and the application's instructions-per-cycle (IPC)?**
1. **Comment on why this is (or is not) the case. Remember to use the data to answer this question, not just intuition and if the data runs counter to your intuition, attempt to use other data to explain why. (Hint: Think about occupancy too)**

#### Hints

There are some existing statistics in the simulator that will help you get the information.
For SIMD Utilization, after each kernel, the simulator print something that looks like:
```bash
Stall:725850»   W0_Idle:4845958»W0_Scoreboard:8325896»  W1:555719»  W2:251750»  W3:171416»  W4:137868»  W5:110052»  W6:80607»   W7:64910»   W8:51295»         W9:44836»   W10:40359»  W11:38322»  W12:36372»  W13:34387»  W14:30871»  W15:31482»  W16:26605»  W17:21532»  W18:16049»  W19:9632»   W20:7005»   W21:      3629»   W22:2292»   W23:1430»   W24:911»W25:708»W26:803»W27:918»W28:886»W29:812»W30:1380»   W31:3317»   W32:413365
```

This print can be parsed as:
```bash
Stall:<If a warp scheduler is stalled on a given cycle +1>»   W0_Idle:<If there is no work to issue in a scheduler (i.e. FE stall) +1>»W0_Scoreboard:<If nothing was issued because of a RAW dependency +1>»  WN:<Warp issued with N lanes active> ...
```

Note that these prints are mutally exclusive, so for each scheduler only one can be true on a given cycle.
These numbers are aggregated over all the kernel launches, so examinging the last print from the .o file will give you the final result from the whole application.
The code that increments these counters lives here:
https://github.com/accel-sim/gpgpu-sim_distribution/blob/1ee03f0116511ac3c2d6ac7688d916191f4f0a6b/src/gpgpu-sim/shader.cc#L1009
and here:
https://github.com/accel-sim/gpgpu-sim_distribution/blob/1ee03f0116511ac3c2d6ac7688d916191f4f0a6b/src/gpgpu-sim/shader.cc#L1423

From this information, you should be able to compute the overall SIMD efficieny. How you do it is up to you,
you can either add an aggregating print to the simulator itself such that it will print something like this after each kernel
completes:
```bash
simd_efficiency=XX%
```

Or you can choose not to modify the simulator at all and simply write a (or modify an existing) post-processing script that computes the SIMD efficieny from the print.
If you choose to do the format (recommended) - you can place the aggregating code and the print in the same place that the original print happens:
https://github.com/accel-sim/gpgpu-sim_distribution/blob/1ee03f0116511ac3c2d6ac7688d916191f4f0a6b/src/gpgpu-sim/shader.cc#L690

For memory divergence, you will need to count both the number of global memory instructions, as well as the number of global memory accesses.
It is left as an exercise for the student to find these.

## Some notes on debugging

When you change the code, you will probably want to debug the simulator (either using prints, trace statements or a debugging/breakpoint tool like gdb).
The best way to do this is to launch an individual app. When run\_simulations.py is called, it creates a directory structure
for launching all the experiments requested.
For example, if I want to debug Rodinia's BFS application, I can do the following:

```bash
# Move to the directory in question
cd ./sim_run_9.1/bfs-rodinia-3.1/__data_graph1MW_6_txt/QV100-PTX-1B_INSN/
```

Inside this directory, all the necessary files are copied and linked to run the application.
A script called justrun.sh is also located here that contains the command you need to run the app.

```bash
qstruct:$ cat justrun.sh
/home/min/a/ee695paa/github/gpu-app-collection/src/..//bin/9.1/release/bfs-rodinia-3.1 ./data/graph1MW_6.txt
```

to run gdb on the simulator, you can simply:
```bash
gdb --args /home/min/a/ee695paa/github/gpu-app-collection/src/..//bin/9.1/release/bfs-rodinia-3.1 ./data/graph1MW_6.txt
```

Since the simulator is loaded as a .so file, if you set a breakpoint before starting, it will say the symbol cannot be found, but it will still work.
For example:

```bash
gdb --args `cat justrun.sh`
GNU gdb (GDB) Red Hat Enterprise Linux 7.6.1-120.el7
Copyright (C) 2013 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.  Type "show copying"
and "show warranty" for details.
This GDB was configured as "x86_64-redhat-linux-gnu".
For bug reporting instructions, please see:
<http://www.gnu.org/software/gdb/bugs/>...
Reading symbols from /home/min/a/ee695paa/github/gpu-app-collection/bin/9.1/release/bfs-rodinia-3.1...(no debugging symbols found)...done.
(gdb) b shader.cc:1423
No symbol table is loaded.  Use the "file" command.
Make breakpoint pending on future shared library load? (y or [n]) y
Breakpoint 1 (shader.cc:1423) pending.
(gdb) r

...

Breakpoint 1, scheduler_unit::cycle (this=0x21b4ea0) at shader.cc:1423
1423      if (!valid_inst)
```

One note on using gdb - the default build of the simulator is with -02, which can make debugging difficult.
To compile the simulator with -O0 (which will make it VERY slow - but much more debuggable)
run:

```bash
source setup_environment.sh debug
make
```

you can switch back and forth using
```bash
source setup_environment.sh release
make
```

If your code is crashing or you want to walk through some stuff that is going gdb is great.
However - if you want to see what is happening on a cycle-by-cycle basis, the simulator's tracing system is more useful.
To enable the tracing system, simply modify the config (gpgpusim.config) file to include:

```bash
# tracing functionality
-trace_enabled 1
-trace_components WARP_SCHEDULER,SCOREBOARD
-trace_sampling_core 0
```

The full list of available traces can be found here:
https://github.com/accel-sim/gpgpu-sim_distribution/blob/dev/src/trace_streams.tup

NOTE: Turning on the traces will create a HUGS amount of output. It is recommended to only do this on an app-by-app basis to understand that is happening and debug. Scaled simulations of all the apps should generally avoid such huge outputs.

Finally, the simulator comes with a detailed cycle-by-cycle visualization tool, called Arielvision.
You will need to have a graphical linux terminal in order to make use of it.

You can turn on the generation of arielvision traces from the cofiguration file using:

```bash
-visualizer_enabled 1
```

This will create a gpgpusim\_visualizer\_\_\*.log.gz file.
This file can then be loaded by the visualizer using the instructions found here:


https://github.com/accel-sim/gpgpu-sim_distribution/blob/dev/aerialvision/README


NOTE: The manual is slightly outdated. To launch the arielvision executable, use the following
and remember that you must do this on a linux machine running the X GUI:

```bash
./gpu-simulator/gpgpu-sim/bin/aerialvision.py
```


## What to hand in

In your assisngment repo - submit a maximum 2-page pdf plotting the results you collected with a few sentences descirbing what
you found with an explanation as to why these differences occurred.

Please create a collect\_stats.sh script that captures the commands you ran
to collect the results you report for the assignment.


