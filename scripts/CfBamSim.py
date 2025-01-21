import argparse
import numpy as np
import pandas as pd
import pysam
import random
import time
import os
import logging
from multiprocessing import Pool

# Function to set up the logger
def setup_logger(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_file = os.path.join(output_dir, "simulation.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

# Function to check if required files exist in the input directory
def check_input_files(file_path, required_files):
    if not os.path.exists(file_path):
        logging.error(f"Input path '{file_path}' does not exist!")
        raise FileNotFoundError(f"Input path '{file_path}' does not exist!")

    for req_file in required_files:
        full_path = os.path.join(file_path, req_file)
        if not os.path.exists(full_path):
            logging.error(f"Required file '{full_path}' not found!")
            raise FileNotFoundError(f"Required file '{full_path}' not found!")

# Function to load hotspot data from file
def load_hotspot_file(hotspot_file):
    if os.path.exists(hotspot_file):
        return pd.read_csv(hotspot_file)
    else:
        logging.warning(f"Hotspot file '{hotspot_file}' not found. Proceeding without hotspots.")
        return pd.DataFrame()

# Function to load annotation data from file
def load_annotation_file(annotation_file):
    if os.path.exists(annotation_file):
        return pd.read_csv(annotation_file, sep='\t', names=['chrom', 'start', 'end', 'type'])
    else:
        logging.warning(f"Annotation file '{annotation_file}' not found. Proceeding without annotations.")
        return pd.DataFrame()

# Function to load length distribution data from file
def load_length_distribution(length_distribution_file):
    if os.path.exists(length_distribution_file):
        return pd.read_csv(length_distribution_file)
    else:
        logging.warning(f"Length distribution file '{length_distribution_file}' not found. Using default distribution.")
        return None

# Function to sample reads based on given distribution
def sample_reads_with_distribution(reads, distribution, num_reads):
    if distribution is None:
        return random.sample(reads, min(len(reads), num_reads))
    lengths = np.random.choice(distribution['length'], size=num_reads, p=distribution['probability'])
    return [read for read in reads if abs(read.template_length) in lengths][:num_reads]

# Function to generate simulated sample
def generate_simulated_sample(args):
    (
        input_interval_file, input_dir, output_dir, simulated_sample_category, sampling_method,
        window_size, disease_ratio_min, disease_ratio_max, target_coverage, target_depth,
        sequencing_type, sample_index, hotspot_file, annotation_file, length_distribution_file
    ) = args

    # Set default disease ratio if not provided
    if disease_ratio_min is None or disease_ratio_max is None:
        if simulated_sample_category == 'normal':
            disease_ratio_min = 0.3
            disease_ratio_max = 1.0
        elif simulated_sample_category == 'diseased':
            disease_ratio_min = 0.01
            disease_ratio_max = 0.2

    logging.info(f"Simulating sample {sample_index} ({simulated_sample_category} category)")
    logging.info(f"disease_ratio_min: {disease_ratio_min}, disease_ratio_max: {disease_ratio_max}")

    # Load bin position and BAM file information
    bin_common_position = pd.read_csv(input_interval_file)
    file_info_path = os.path.join(input_dir, 'bam_file_info.txt')
    file_info = pd.read_csv(file_info_path, sep='\t', names=['filename', 'category'])

    if file_info.empty:
        logging.error(f"BAM file info '{file_info_path}' is empty!")
        raise ValueError(f"BAM file info '{file_info_path}' is empty!")

    NPsamfiles = {}
    CTsamfiles = {}

    # Read BAM files
    for index, row in file_info.iterrows():
        file_path = os.path.join(input_dir, row['filename'])
        try:
            if not os.path.exists(file_path):
                logging.warning(f"BAM file '{file_path}' does not exist. Skipping.")
                continue

            if row['category'] == 'normal':
                NPsamfiles[row['filename']] = pysam.AlignmentFile(file_path, "rb")
            elif row['category'] == 'diseased':
                CTsamfiles[row['filename']] = pysam.AlignmentFile(file_path, "rb")
        except OSError as e:
            logging.warning(f"Skipping file '{file_path}' due to error: {e}")
            continue

    # Select base and mix samples for normal or diseased category
    if simulated_sample_category == 'normal':
        if len(NPsamfiles) < 2:
            logging.error("At least two normal BAM files are required for normal sample simulation!")
            raise ValueError("At least two normal BAM files are required for normal sample simulation!")
        base_sample = random.choice(list(NPsamfiles.values()))
        mix_sample = random.choice([file for file in NPsamfiles.values() if file != base_sample])
    elif simulated_sample_category == 'diseased':
        if len(NPsamfiles) < 2 or len(CTsamfiles) < 2:
            logging.error("At least two normal and two diseased BAM files are required for diseased sample simulation!")
            raise ValueError("At least two normal and two diseased BAM files are required for diseased sample simulation!")
        base_sample = random.choice(list(NPsamfiles.values()))
        mix_sample = random.choice(list(CTsamfiles.values()))

    # Load other files like hotspots, annotations, and length distributions
    hotspots = load_hotspot_file(hotspot_file)
    annotations = load_annotation_file(annotation_file)
    length_distribution = load_length_distribution(length_distribution_file)

    # Create new BAM file for the simulated sample
    new_sample_path = os.path.join(output_dir, f"Simulated_{simulated_sample_category}_sample_{sample_index}.bam")
    new_file = pysam.AlignmentFile(new_sample_path, "wb", template=base_sample)

    start_time = time.time()

    # Iterate over the reference sequences
    for ref in base_sample.references:
        ref_length = base_sample.get_reference_length(ref)
        bins = range(1, ref_length, window_size)

        for bin_start in bins:
            bin_end = min(bin_start + window_size - 1, ref_length)

            CT_ratio = random.uniform(disease_ratio_min, disease_ratio_max)
            NP_ratio = 1 - CT_ratio

            # Fetch reads from the reference bin
            reads = list(base_sample.fetch(ref, bin_start, bin_end))
            reads_mix = list(mix_sample.fetch(ref, bin_start, bin_end))

            # Sample the reads according to the distribution
            sampled_reads_base = sample_reads_with_distribution(reads, length_distribution, int(len(reads) * NP_ratio))
            sampled_reads_mix = sample_reads_with_distribution(reads_mix, length_distribution, int(len(reads_mix) * CT_ratio))

            for read in sampled_reads_base + sampled_reads_mix:
                # Only write reads that have a mate in paired-end sequencing
                new_file.write(read)
                if sequencing_type == "paired-end":
                    try:
                        mate_read = base_sample.mate(read)
                        new_file.write(mate_read)
                    except ValueError:
                        logging.warning(f"Failed to find mate for read: {read.query_name}")
                        # If no mate is found, skip writing this read
                        continue

    new_file.close()
    end_time = time.time()
    logging.info(f"Sample {sample_index} simulation completed in {end_time - start_time:.2f} seconds")

# Main function to set up the simulation and handle multiprocessing
def main(args):
    setup_logger(args.output_dir)
    required_files = ['bam_file_info.txt']
    check_input_files(args.input_dir, required_files)

    pool_args = [
        (
            args.input_interval_file, args.input_dir, args.output_dir, args.simulated_sample_category,
            args.sampling_method, args.window_size, args.disease_ratio_min, args.disease_ratio_max,
            args.target_coverage, args.target_depth, args.sequencing_type, i,
            args.hotspot_file, args.annotation_file, args.length_distribution_file
        )
        for i in range(1, args.num_simulated_samples + 1)
    ]

    with Pool(args.num_threads) as pool:
        pool.map(generate_simulated_sample, pool_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate simulated cfDNA methylation sequencing data with multithreading.')
    parser.add_argument('--input_interval_file', type=str, required=True, help='Path to the input interval file')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input bam files and bam_file_info.txt')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output simulated samples')
    parser.add_argument('--simulated_sample_category', type=str, default='diseased', choices=['normal', 'diseased'],
                        help='Category of the simulated sample')
    parser.add_argument('--sampling_method', type=str, default='uniform', choices=['uniform', 'non-uniform'], help='Sampling method')
    parser.add_argument('--window_size', type=int, default=20000, help='Size of the window')
    parser.add_argument('--disease_ratio_min', type=float, help='Minimum ratio of disease reads in a bin')
    parser.add_argument('--disease_ratio_max', type=float, help='Maximum ratio of disease reads in a bin')
    parser.add_argument('--target_coverage', type=float, help='Target genome coverage for the simulated bam files')
    parser.add_argument('--target_depth', type=int, help='Target sequencing depth for the simulated bam files')
    parser.add_argument('--num_simulated_samples', type=int, default=2, help='Number of simulated samples')
    parser.add_argument('--num_threads', type=int, default=2, help='Number of threads to use for simulation')
    parser.add_argument('--sequencing_type', type=str, default='single-end', choices=['single-end', 'paired-end'],
                        help='Type of sequencing to simulate')
    parser.add_argument('--hotspot_file', type=str, default='', help='Path to the hotspot file (optional)')
    parser.add_argument('--annotation_file', type=str, default='', help='Path to the annotation file (optional)')
    parser.add_argument('--length_distribution_file', type=str, default='', help='Path to the length distribution file (optional)')

    args = parser.parse_args()
    main(args)
