import multiprocessing
import os
import sys
import time
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from prefixspan import PrefixSpan
import cupy as cp  # Use cupy for GPU acceleration

# Increase recursion limit
sys.setrecursionlimit(10000)

def create_directory(path):
    """Create a directory if it does not exist"""
    os.makedirs(path, exist_ok=True)

def load_unikmer_map(unikmer_file, debug_dir):
    """Load unikmer mapping from a file"""
    unikmer_map = {}
    for counter, line in enumerate(unikmer_file):
        unikmer_map[line.strip()] = counter

    # Save unikmer_map to a file for debugging
    debug_unikmer_map_path = os.path.join(debug_dir, "debug_unikmer_map.txt")
    with open(debug_unikmer_map_path, "w") as f:
        for unikmer, idx in unikmer_map.items():
            f.write(f"Unikmer: {unikmer}, Index: {idx}\n")

    return unikmer_map

def save_unikmer_map(unikmer_map, output_file):
    """Save unikmer mapping to a file"""
    with open(output_file, "w") as out_file:
        for unikmer, idx in unikmer_map.items():
            out_file.write(f"{unikmer},{idx}\n")

def process_reads_and_unikmers(unikmer_map, read_file, k):
    """Process the read file and generate read-to-unikmer and unikmer-to-read mappings"""
    read_to_unikmer_map = defaultdict(dict)
    unikmer_to_read_map = defaultdict(list)
    for record in read_file:
        read = record.seq
        read_id = record.id
        for i in range(len(read) - k + 1):
            kmer = str(read[i:i + k])
            rev_kmer = str(read[i:i + k].reverse_complement())
            for kmer_variant in [kmer, rev_kmer]:
                if kmer_variant in unikmer_map:
                    unikmer_id = unikmer_map[kmer_variant]
                    read_to_unikmer_map[read_id][unikmer_id] = i
                    unikmer_to_read_map[unikmer_id].append((read_id, i))
    return read_to_unikmer_map, unikmer_to_read_map

def save_mappings(read_to_unikmer_map, unikmer_to_read_map, out_r2u, out_u2r):
    """Save the read-to-unikmer and unikmer-to-read mappings to files"""
    with open(out_r2u, "w") as r2u_file, open(out_u2r, "w") as u2r_file:
        for read, unikmers in read_to_unikmer_map.items():
            r2u_file.write(f"{read}>")
            for unikmer, pos in unikmers.items():
                r2u_file.write(f"({unikmer},{pos});")
            r2u_file.write("\n")
        for unikmer, reads in unikmer_to_read_map.items():
            u2r_file.write(f"{unikmer}>")
            for read, pos in reads:
                u2r_file.write(f"({read},{pos});")
            u2r_file.write("\n")

def find_frequent_patterns_parallel_gpu(read_to_unikmer_map, min_support, max_pattern_length, num_threads=12, gpu_id=0, debug_dir=None):
    """Find frequent patterns using GPU acceleration"""
    
    # Convert sequences to numeric representations (for example, integer encoding of the k-mers)
    sequences = [[unikmer for unikmer in read_to_unikmer_map[read].keys()] for read in read_to_unikmer_map]

    # Save sequences to a file for debugging
    if debug_dir:
        debug_sequences_path = os.path.join(debug_dir, "debug_sequences.txt")
        with open(debug_sequences_path, "w") as f:
            for seq in sequences:
                f.write(f"{seq}\n")

    # Determine the maximum sequence length
    max_len = max(len(seq) for seq in sequences)

    # Padding sequences to the maximum length with zeros
    sequences_padded = [seq + [0] * (max_len - len(seq)) for seq in sequences]

    # Save sequences_padded to a file for debugging
    if debug_dir:
        debug_sequences_padded_path = os.path.join(debug_dir, "debug_sequences_padded.txt")
        with open(debug_sequences_padded_path, "w") as f:
            for seq in sequences_padded:
                f.write(f"{seq}\n")

    # Split sequences into chunks for parallel processing
    chunk_size = len(sequences) // num_threads
    chunks = [sequences_padded[i:i + chunk_size] for i in range(0, len(sequences), chunk_size)]

    # Use ProcessPoolExecutor to parallelize the CPU-bound task
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_chunk, chunk, min_support, max_pattern_length, gpu_id, debug_dir) for chunk in chunks]
        frequent_patterns = []
        for future in as_completed(futures):
            frequent_patterns.extend(future.result())

    # Save frequent_patterns to a file for debugging
    if debug_dir:
        debug_frequent_patterns_path = os.path.join(debug_dir, "debug_frequent_patterns.txt")
        with open(debug_frequent_patterns_path, "w") as f:
            for support, pattern in frequent_patterns:
                f.write(f"Support: {support}, Pattern: {pattern}\n")

    # Use GPU to count the support of patterns (Note: Counting patterns needs to be done on CPU for now)
    pattern_counts = defaultdict(int)
    for support, pattern in frequent_patterns:
        pattern_counts[tuple(pattern)] += support

    # Save pattern_counts to a file for debugging
    if debug_dir:
        debug_pattern_counts_path = os.path.join(debug_dir, "debug_pattern_counts.txt")
        with open(debug_pattern_counts_path, "w") as f:
            for pattern, count in pattern_counts.items():
                f.write(f"Pattern: {pattern}, Count: {count}\n")

    # Return the pattern counts as a dictionary
    return dict(pattern_counts)

def process_chunk(chunk, min_support, max_pattern_length, gpu_id, debug_dir=None):
    """Process a chunk of sequences in a subprocess"""
    import cupy as cp
    cp.cuda.Device(gpu_id).use()  # Initialize GPU context in the subprocess

    # Convert sequences to cupy arrays with integer type
    sequences_gpu = cp.array(chunk, dtype=cp.int32)  # Use integer encoding for k-mers

    # Save sequences_gpu to a file for debugging
    if debug_dir:
        debug_sequences_gpu_path = os.path.join(debug_dir, "debug_sequences_gpu.txt")
        with open(debug_sequences_gpu_path, "w") as f:
            f.write(f"{sequences_gpu}\n")

    # Extract frequent patterns using PrefixSpan
    ps = PrefixSpan(chunk)
    frequent_patterns = ps.frequent(min_support)
    frequent_patterns = [(support, pattern) for support, pattern in frequent_patterns if len(pattern) <= max_pattern_length]

    # Save frequent_patterns to a file for debugging
    if debug_dir:
        debug_frequent_patterns_path = os.path.join(debug_dir, "debug_frequent_patterns.txt")
        with open(debug_frequent_patterns_path, "w") as f:
            for support, pattern in frequent_patterns:
                f.write(f"Support: {support}, Pattern: {pattern}\n")

    return frequent_patterns

def cluster_reads_by_patterns(read_to_unikmer_map, frequent_patterns, debug_dir=None):
    """Cluster reads based on frequent patterns"""
    clusters = defaultdict(list)
    unclassified_reads = []  # Store unclassified reads

    for read, unikmers in read_to_unikmer_map.items():
        unikmer_ids = set(unikmers.keys())
        matched = False
        for pattern in frequent_patterns:
            if all(p in unikmer_ids for p in pattern):
                clusters[tuple(pattern)].append(read)
                matched = True
                break  # A read can only belong to one cluster
        if not matched:
            unclassified_reads.append(read)  # Add to unclassified reads

    # Add unclassified reads to clusters
    if unclassified_reads:
        clusters["unclassified"] = unclassified_reads

    # Save clusters to a file for debugging
    if debug_dir:
        debug_clusters_path = os.path.join(debug_dir, "debug_clusters.txt")
        with open(debug_clusters_path, "w") as f:
            for pattern, reads in clusters.items():
                f.write(f"Pattern: {pattern}, Reads: {reads}\n")

    return clusters

def save_clusters(clusters, cluster_file):
    """Save the clustering results to a file"""
    with open(cluster_file, "w") as out_file:
        out_file.write(f"{len(clusters)}\n")
        for pattern, reads in clusters.items():
            out_file.write(f"Pattern: {pattern}, Reads: {','.join(map(str, reads))}\n")

def write_fasta_from_clusters(clusters, read_file, output_dir, debug_dir=None):
    """Write the clustered sequences into multiple FASTA files, with unclassified reads saved separately."""
    read_dict = {record.id: record.seq for record in read_file}

    # Save read_dict to a file for debugging
    if debug_dir:
        debug_read_dict_path = os.path.join(debug_dir, "debug_read_dict.txt")
        with open(debug_read_dict_path, "w") as f:
            for read_id, seq in read_dict.items():
                f.write(f"Read ID: {read_id}, Sequence: {seq}\n")

    # Save unclassified reads to a separate file
    if "unclassified" in clusters:
        unclassified_reads = clusters["unclassified"]
        unclassified_sequences = [SeqRecord(Seq(read_dict[str(read)]), id=f"unclassified_read_{read}") for read in unclassified_reads]
        SeqIO.write(unclassified_sequences, os.path.join(output_dir, "unclassified.fasta"), "fasta")
        del clusters["unclassified"]  # Remove unclassified reads from clusters to avoid duplication

    # Save clustered reads to their respective files
    for idx, (pattern, cluster_reads) in enumerate(clusters.items()):
        sequences = [SeqRecord(Seq(read_dict[str(read)]), id=f"cluster_{idx}_read_{read}") for read in cluster_reads]
        SeqIO.write(sequences, os.path.join(output_dir, f"cluster_{idx}.fasta"), "fasta")

def run_rambler(args):
    """Execute the Rambler clustering process"""
    print("Initializing Rambler...")
    create_directory(args.output + "/assembly/hifiasm")
    unikmer_file = open(args.unikmers, "r")
    read_file1 = SeqIO.parse(args.reads, "fasta")
    read_file2 = SeqIO.parse(args.reads, "fasta")

    # Define debug directory (same as unikmer_map.txt's directory)
    debug_dir = os.path.join(args.output, "intermediates")

    # Load unikmer mapping
    unikmer_map = load_unikmer_map(unikmer_file, debug_dir)
    save_unikmer_map(unikmer_map, os.path.join(debug_dir, "unikmer_map.txt"))
    
    # Process reads and unikmers, generate mappings
    r2u_map, u2r_map = process_reads_and_unikmers(unikmer_map, read_file1, args.kmer_size)
    save_mappings(r2u_map, u2r_map, os.path.join(debug_dir, "read_to_unikmer_map.txt"), 
                  os.path.join(debug_dir, "unikmer_to_read_map.txt"))
    
    print("Unikmer mapping completed.")
    
    # Find frequent patterns (using GPU acceleration)
    frequent_patterns = find_frequent_patterns_parallel_gpu(r2u_map, args.min_support, max_pattern_length=5, gpu_id=args.gpu_id, debug_dir=debug_dir)
    
    # Cluster reads based on frequent patterns
    clusters = cluster_reads_by_patterns(r2u_map, frequent_patterns, debug_dir)

    # Save clustering results and sequences
    save_clusters(clusters, os.path.join(args.output, "clusters", "cluster_log.txt"))
    write_fasta_from_clusters(clusters, read_file2, os.path.join(args.output, "clusters"), debug_dir)
    
    print("Clustering completed.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  # 设置启动方法为 spawn
    parser = argparse.ArgumentParser(description="Rambler: Read clustering based on frequent k-mer patterns.")
    parser.add_argument("-r", "--reads", required=True, help="Path to the reads file in FASTA format.")
    parser.add_argument("-u", "--unikmers", required=True, help="Path to the unique k-mers file.")
    parser.add_argument("-o", "--output", required=True, help="Path to the output directory.")
    parser.add_argument("-k", "--kmer_size", type=int, required=True, help="Size of the k-mers.")
    parser.add_argument("-s", "--min_support", type=int, required=True, help="Minimum support for frequent patterns.")
    parser.add_argument("-g", "--gpu_id", type=int, required=True, help="GPU ID to use (0, 1, 2, or 3).")
    
    args = parser.parse_args()
    run_rambler(args)