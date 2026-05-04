# Sampling Report

Approach: stratified downsampling by `rooms_count`.

Reason: 1-room listings dominate the current CIAN snapshot. Balancing by room
segment prevents baseline/model experiments from being overly optimized for the
largest segment only.

Original rows: 1304

Original room distribution:

rooms_count
0    264
1    260
2    262
3    255
4    263

Balanced rows: 1275

Balanced room distribution:

rooms_count
0    255
1    255
2    255
3    255
4    255
