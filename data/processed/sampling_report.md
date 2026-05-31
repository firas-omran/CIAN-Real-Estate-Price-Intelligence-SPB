# Sampling Report

Approach: stratified downsampling by `rooms_count`.

Reason: 1-room listings dominate the current CIAN snapshot. Balancing by room
segment prevents baseline/model experiments from being overly optimized for the
largest segment only.

Original rows: 2202

Original room distribution:

rooms_count
0    423
1    480
2    454
3    391
4    454

Balanced rows: 1955

Balanced room distribution:

rooms_count
0    391
1    391
2    391
3    391
4    391
