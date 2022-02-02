import torch
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


class Splade(torch.nn.Module):

    def __init__(self, model_type_or_dir, agg="max"):
        super().__init__()
        self.transformer = AutoModelForMaskedLM.from_pretrained(model_type_or_dir)
        assert agg in ("sum", "max")
        self.agg = agg

    def forward(self, **kwargs):
        with torch.cuda.amp.autocast():
            out = self.transformer(**kwargs)["logits"]  # output (logits) of MLM head, shape (bs, pad_len, voc_size)
            if self.agg == "max":
                values, _ = torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
                return values
                # 0 masking also works with max because all activations are positive
            else:
                return torch.sum(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)


class CollectionDatasetPreLoad:
    """
    dataset to iterate over a document/query collection, format per line: format per line: doc_id \t doc
    """

    def __init__(self, data_path):
        self.data_path = data_path

        self.data_dict = {}  # => dict that maps the id to the line offset (position of pointer in the file)
        self.line_dict = {}  # => dict that maps the id to the line offset (position of pointer in the file)
        print("Preloading dataset", flush=True)
        with open(self.data_path) as reader:
            for i, line in enumerate(tqdm(reader)):
                if len(line) > 1:
                    id_, *data = line.split("\t")  # first column is id
                    data = " ".join(" ".join(data).splitlines())
                    self.data_dict[i] = data
                    self.line_dict[i] = id_.strip()

        self.nb_ex = len(self.data_dict)

    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        return self.line_dict[idx], self.data_dict[idx]


class DataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(self, tokenizer_path, max_length, **kwargs):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        super().__init__(collate_fn=self.collate_fn, **kwargs, pin_memory=True)

    def collate_fn(self, batch):
        raise NotImplementedError("must implement this method")

    def collate_fn(self, batch):
        """
        batch is a list of tuples, each tuple has 2 (text) items (id_, doc)
        """
        id_, d = zip(*batch)
        processed_passage = self.tokenizer(list(d),
                                           add_special_tokens=True,
                                           padding="longest",  # pad to max sequence length in batch
                                           truncation="longest_first",  # truncates to max model length,
                                           max_length=self.max_length,
                                           return_attention_mask=True)
        return {**{k: torch.tensor(v) for k, v in processed_passage.items()},
                "id": torch.tensor([int(i) for i in id_], dtype=torch.long),
                "text": d}
