[2595] python3.12@127.0.0.1
  fused_attn_kernel_Kt1_A4f16_7v8l0_8t1_ci16_p16_A4f16_7v8l0_8t1_ci16_p16_A4f16_7v8l0_8t1_ci16_p16_A4f16_7v8l0_8t1_ci16_p16_Sf32_I8_I64_I64_I64 (128, 16, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Tile Statistics
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Size (Chosen by Compiler)                      256
    Cluster Size                                           0
    Launch Execution Model                       Tile (Tile)
    #Thread Blocks Launched                             2048
    #Tile Blocks Specified                              2048
    Blocks as Clusters                                     0
    ------------------------------- ----------- ------------

    INF   One thread block is mapped to execute one tile block instructions.                                            

    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz        12.47
    SM Frequency                    Ghz         2.27
    Elapsed Cycles                cycle      1596064
    Memory Throughput                 %        38.26
    DRAM Throughput                   %         4.51
    Duration                         us       700.38
    L1/TEX Cache Throughput           %        28.87
    L2 Cache Throughput               %        38.26
    SM Active Cycles              cycle   1427759.78
    Compute (SM) Throughput           %        45.21
    ----------------------- ----------- ------------

    OPT   This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak           
          performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak           
          typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potential       
          reasons.                                                                                                      

    Section: GPU Speed Of Light Roofline Chart (Overview)
    INF   The ratio of peak float (FP32) to double (FP64) performance on this device is 64:1. The workload achieved 11% 
          of this device's FP32 peak performance and 0% of its FP64 peak performance. See the Profiling Guide           
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) for more details on roofline      
          analysis.                                                                                                     

    Section: PM Sampling
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Maximum Buffer Size             Mbyte        65.73
    Maximum Sampling Interval          us            3
    # Pass Groups                                    2
    ------------------------- ----------- ------------

    Section: Compute Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Executed Ipc Active   inst/cycle         1.30
    Executed Ipc Elapsed  inst/cycle         1.16
    Issue Slots Busy               %        29.12
    Issued Ipc Active     inst/cycle         1.30
    SM Busy                        %        45.21
    -------------------- ----------- ------------

    Section: Memory Workload Analysis
    --------------------------------------- ----------- ------------
    Metric Name                             Metric Unit Metric Value
    --------------------------------------- ----------- ------------
    Local Memory Spilling Requests                                 0
    Local Memory Spilling Request Overhead            %            0
    L2 Sector Promotion Misses                        %            0
    Shared Memory Spilling Requests                                0
    Shared Memory Spilling Request Overhead           %            0
    Memory Throughput                           Gbyte/s        71.91
    Mem Busy                                          %        38.26
    Max Bandwidth                                     %        37.96
    L1/TEX Hit Rate                                   %            0
    L2 Persisting Size                            Mbyte        25.17
    L2 Compression Success Rate                       %            0
    L2 Compression Ratio                                           0
    L2 Compression Input Sectors                 sector            0
    L2 Hit Rate                                       %        96.95
    Mem Pipes Busy                                    %        15.62
    --------------------------------------- ----------- ------------

    Section: Scheduler Statistics
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %        32.57
    Issued Warp Per Scheduler                        0.33
    No Eligible                            %        67.43
    Active Warps Per Scheduler          warp         1.50
    Eligible Warps Per Scheduler        warp         0.33
    ---------------------------- ----------- ------------

    OPT   Est. Local Speedup: 54.79%                                                                                    
          Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only    
          issues an instruction every 3.1 cycles. This might leave hardware resources underutilized and may lead to     
          less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average    
          of 1.50 active warps per scheduler, but only an average of 0.33 warps were eligible per cycle. Eligible       
          warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no      
          eligible warp results in no instruction being issued and the issue slot remains unused. To increase the       
          number of eligible warps, reduce the time the active warps are stalled by inspecting the top stall reasons    
          on the Warp State Statistics and Source Counters sections.                                                    

    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle         4.59
    Warp Cycles Per Executed Instruction           cycle         4.60
    Avg. Active Threads Per Warp                                32.00
    Avg. Not Predicated Off Threads Per Warp                    31.36
    ---------------------------------------- ----------- ------------

    Section: Instruction Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Local Memory Spilling Requests                  byte            0
    Shared Memory Spilling Requests                 byte            0
    Avg. Executed Instructions Per Scheduler        inst    462269.71
    Executed Instructions                           inst    347626823
    Avg. Issued Instructions Per Scheduler          inst    463410.84
    Issued Instructions                             inst    348484954
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 5.01%                                                                                           
          This workload executes 74227712 fused and 107814912 non-fused FP32 instructions. By converting pairs of       
          non-fused instructions to their fused                                                                         
          (https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point), higher-throughput equivalent, the     
          achieved FP32 performance could be increased by up to 30% (relative to its current performance).              

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Cluster Scheduling Policy                           PolicySpread
    Cluster Size                                                   0
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                   2048
    Preferred Cluster Size                                         0
    Registers Per Thread             register/thread             255
    Shared Memory Configuration Size           Kbyte           65.54
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block       Kbyte/block           32.83
    # SMs                                         SM             188
    Stack Size                                                  1024
    Threads                                   thread          524288
    # TPCs                                                        94
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                               10.89
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Max Active Clusters                 cluster            0
    Max Cluster Size                      block            8
    Overall GPU Occupancy                     %            0
    Cluster Occupancy                         %            0
    Block Limit Barriers                  block           24
    Block Limit SM                        block           24
    Block Limit Registers                 block            1
    Block Limit Shared Mem                block            1
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp            8
    Theoretical Occupancy                     %        16.67
    Achieved Occupancy                        %        12.42
    Achieved Active Warps Per SM           warp         5.96
    ------------------------------- ----------- ------------

    OPT   Est. Speedup: 54.79%                                                                                          
          The 2.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 12. This kernel's theoretical occupancy (16.7%) is limited by the number of required      
          registers, and the required amount of shared memory.                                                          

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle       393452
    Total DRAM Elapsed Cycles        cycle    139721728
    Average L1 Active Cycles         cycle   1427759.78
    Total L1 Elapsed Cycles          cycle    299191414
    Average L2 Active Cycles         cycle   1249533.16
    Total L2 Elapsed Cycles          cycle     89791040
    Average SM Active Cycles         cycle   1427759.78
    Total SM Elapsed Cycles          cycle    299191414
    Average SMSP Active Cycles       cycle   1422699.32
    Total SMSP Elapsed Cycles        cycle   1196765656
    -------------------------- ----------- ------------

    OPT   Est. Speedup: 8.977%                                                                                          
          One or more SMs have a much higher number of active cycles than the average number of active cycles. Maximum  
          instance value is 10.01% above the average, while the minimum instance value is 6.97% below the average.      
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 8.882%                                                                                          
          One or more SMSPs have a much higher number of active cycles than the average number of active cycles.        
          Maximum instance value is 9.94% above the average, while the minimum instance value is 7.14% below the        
          average.                                                                                                      
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 8.977%                                                                                          
          One or more L1 Slices have a much higher number of active cycles than the average number of active cycles.    
          Maximum instance value is 10.01% above the average, while the minimum instance value is 6.97% below the       
          average.                                                                                                      

    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.02
    Branch Instructions              inst      5782115
    Branch Efficiency                   %        99.95
    Avg. Divergent Branches      branches         2.72
    ------------------------- ----------- ------------

