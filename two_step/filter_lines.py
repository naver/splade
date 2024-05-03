import argparse
from tqdm import tqdm 
def copy_filter_line(file_a: str, file_b: str):
    with open(file_a, 'r') as file_a_obj, open(file_b, 'w') as file_b_obj:
        for line in tqdm(file_a_obj):
            qid, q0, did, pos, score, name = line.strip().split('\t')
            if qid == did:
                continue
            else:
                file_b_obj.write(line)




# Example usage
parser = argparse.ArgumentParser()
parser.add_argument("file_a", help="first file path")
parser.add_argument("file_b", help="second file path")
args = parser.parse_args()
copy_filter_line(args.file_a, args.file_b)
