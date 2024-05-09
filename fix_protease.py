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
            lines = file.readlines()
            header = lines[0].strip()
            sequence = ''.join(line.strip() for line in lines[1:])

        modified_sequence = modify_fasta_sequence(sequence, modifications)
        output_content = header + '\n' + modified_sequence

        # Define the new file path with appended suffix
        new_file_path = output_dir / f'{fasta_file.stem}{suffix}.fasta'
        with open(new_file_path, 'w') as file:
            file.write(output_content)
        print(f'Processed {fasta_file.name} to {new_file_path.name}')

def main():
    parser = argparse.ArgumentParser(description='Modify FASTA sequences and save them to a new directory.')
    parser.add_argument('--input_dir', type=str, help='Directory containing FASTA files')
    parser.add_argument('--output_dir', type=str, help='Directory to save modified FASTA files')
    parser.add_argument('--modifications', type=str, help='Modifications in the format "A1,B2,C3"')
    parser.add_argument('--suffix', type=str, default='_modified', help='Suffix to append to file names')

    args = parser.parse_args()
    process_files(args.input_dir, args.output_dir, args.modifications, args.suffix)

if __name__ == '__main__':
    main()
