import argparse
from pathlib import Path

def modify_fasta_sequence(sequence, modifications):
    modifications_list = modifications.split(',')
    for mod in modifications_list:
        aa = mod[0]  # The amino acid to replace with
        pos = int(mod[1:])  # The position to replace at, 1-based index
        sequence = sequence[:pos-1] + aa + sequence[pos:]
    return sequence

def process_files(input_dir, output_dir, modifications, suffix):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)  # Create the output directory if it does not exist

    for fasta_file in input_dir.glob('*.fasta'):
        with open(fasta_file, 'r') as file:
            sequences = []
            current_sequence = []
            header = None
            for line in file:
                if line.startswith('>'):  # Start of a new sequence
                    if header and current_sequence:  # Save the previous sequence
                        sequences.append((header, ''.join(current_sequence)))
                    # Append suffix to the protein ID in the header
                    header = line.strip() + suffix
                    current_sequence = []  # Reset sequence list
                else:
                    current_sequence.append(line.strip())
            if header and current_sequence:  # Save the last sequence if there is one
                sequences.append((header, ''.join(current_sequence)))

        # Process each sequence and write to the new file
        new_file_path = output_dir / f'{fasta_file.stem}{suffix}.fasta'
        with open(new_file_path, 'w') as new_file:
            for header, sequence in sequences:
                modified_sequence = modify_fasta_sequence(sequence, modifications)
                output_content = f'{header}\n{modified_sequence}\n'
                new_file.write(output_content)
        print(f'Processed {fasta_file.name} to {new_file_path.name}')

def main():
    parser = argparse.ArgumentParser(description='Modify FASTA sequences and save them to a new directory.')
    parser.add_argument('--input_dir', type=str, help='Directory containing FASTA files')
    parser.add_argument('--output_dir', type=str, help='Directory to save modified FASTA files')
    parser.add_argument('--modifications', type=str, help='Modifications in the format "A1,B2,C3"')
    parser.add_argument('--suffix', type=str, default='_modified', help='Suffix to append to file names and headers')

    args = parser.parse_args()
    process_files(args.input_dir, args.output_dir, args.modifications, args.suffix)

if __name__ == '__main__':
    main()
