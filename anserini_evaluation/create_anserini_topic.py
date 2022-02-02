import argparse
import os

import numpy as np
import torch
import utils
from tqdm.auto import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--topic_batch_size", type=int)
    parser.add_argument("--input_topic_path", type=str)
    parser.add_argument("--output_topic_path", type=str)
    parser.add_argument("--output_topic_name", type=str)
    parser.add_argument("--splade_weights_path", type=str)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--quantization_factor", type=int, default=100)
    args = parser.parse_args()

    d_collection = utils.CollectionDatasetPreLoad(data_path=args.input_topic_path)
    d_loader = utils.DataLoader(dataset=d_collection, tokenizer_path=args.splade_weights_path,
                                max_length=args.max_length, batch_size=args.topic_batch_size,
                                shuffle=False, num_workers=10)
    is_cuda = torch.cuda.is_available()
    model = utils.Splade(args.splade_weights_path)
    if is_cuda:
        model = torch.nn.DataParallel(model).cuda()

    vocab_dict = d_loader.tokenizer.get_vocab()
    vocab_dict = {v: k for k, v in vocab_dict.items()}
    os.makedirs(args.output_topic_path, exist_ok=True)
    collection_file = open(os.path.join(args.output_topic_path, "{}.tsv".format(args.output_topic_name)), "w")
    n = len(d_collection)
    with torch.no_grad():
        for t, batch in enumerate(tqdm(d_loader)):
            inputs = {k: v for k, v in batch.items() if k not in {"id", "text"}}
            if is_cuda:
                for k, v in inputs.items():
                    inputs[k] = v.cuda()
            batch_rep = model(**inputs).cpu().numpy()
            for local, (rep, id_, text) in enumerate(zip(batch_rep, batch["id"], batch["text"])):
                id_ = id_.item()
                idx = np.nonzero(rep)
                # then extract values:
                data = rep[idx]
                data = np.rint(data * args.quantization_factor).astype(int)
                dict_splade = dict()
                for id_token, value_token in zip(idx[0], data):
                    if value_token > 0:
                        real_token = vocab_dict[id_token]
                        dict_splade[real_token] = int(value_token)
                if len(dict_splade.keys()) == 0:
                    print("empty input =>", id_)
                    dict_splade[
                        vocab_dict[998]] = 1  # in case of empty doc we fill with "[unused993]" token (just to fill
                    # and avoid issues with anserini), in practice happens just a few times ...
                string_splade = " ".join(
                    [" ".join([str(real_token)] * freq) for real_token, freq in dict_splade.items()])
                collection_file.write(str(id_) + "\t" + string_splade + "\n")

    print("done iterating over the topics...", flush=True)
