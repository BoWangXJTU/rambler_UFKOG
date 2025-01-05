# RAmbler_UFKOG
This repository is a revision of RAmbler. Based on SUNK (Singly Unique Nucleotide k-mer), the UFKOG (Unique Frequent k-mer Order Group) is developed using the PrefixScan sequence pattern mining algorithm. RAmbler_UFKOG is used to mine multiple subtle sequence group differences within longer sequence ranges, enabling precise differentiation of highly repetitive sequence units. This can be helpful for further Whole Genome Duplication (WGD) genome assembly.

# Path set
export PATH=~/miniconda3/envs/hifiasm0.24.0/bin:$PATH
export PATH=~/miniconda3/envs/cupy/bin:$PATH
export CUDA_HOME=/usr/local/cuda11.1
export PATH=/usr/local/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH

# Example run
`bash run_UFKOG_v5-3_gpu.sh -r ../data/reads.fa -u ../data/ukmer.fa -o ../output`  

The output format of patterns counts:
Pattern: (4, 5), Count: 14
Pattern: (4, 5, 6), Count: 14
Pattern: (4, 5, 6, 7), Count: 14
Pattern: (5, 6), Count: 14
Pattern: (5, 6, 7), Count: 14
Pattern: (0, 1, 2), Count: 16

The output format of frequent patterns:
Support: 1, Pattern: [4, 5]
Support: 1, Pattern: [4, 5, 6]
Support: 1, Pattern: [4, 5, 6, 7]
Support: 1, Pattern: [5, 6]
Support: 1, Pattern: [5, 6, 7]
Support: 1, Pattern: [0, 1, 2]

Users can find the reads files clustered based on frequent patterns in the "clusters" folder. For example, if there are two categories in the results, assembling them will yield two genome sequences after WGD. The other steps are the same as RAmbler.

# TODO
GPU acceleration for real data.
