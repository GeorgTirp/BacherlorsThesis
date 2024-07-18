# BacherlorsThesis
The files uploaded here are the python files used in my thesis.
The code is based on the AlgoPerf self tuning jax baseline or adamw (https://github.com/mlcommons/algorithmic-efficiency/tree/main/prize_qualification_baselines/self_tuning)
The code is run like any self tuning submission for the Algoperf Benchmark. To set up the environment and run the code see the tutorials provided on the github for the Benchmark:
https://github.com/mlcommons/algorithmic-efficiency

I also added the visualize_runs.py I sued to plot my data

TO BE FIXED:
-You can currently not continue training on a workload due to errors when saving the validation metrics
-Not sure if the reset for the big workloads (imagenet and librispeech) lead to OOM KILL due to loading in the momentum terms
