import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModelForSeq2SeqLM,T5EncoderModel,MT5EncoderModel
from transformers.modeling_outputs import SequenceClassifierOutput

# try:
#     from transformers.adapters import ParallelConfig,AdapterConfig
# except:
#     print("No adapters")

from splade.tasks.amp import NullContextManager


class TransformerRank(torch.nn.Module):
    def __init__(self, model_type_or_dir, model_type_or_dir_q=None,fp16=True, bf16=False,force_nofp=False, num_labels=1):
        """
        model_type_or_dir is either the name of a pre-trained model (e.g. bert-base-uncased), or the path to
        directory containing model weights, vocab etc.
        """
        super().__init__()
        assert model_type_or_dir_q is None
        self.config = AutoConfig.from_pretrained(model_type_or_dir)
        self.config.num_labels = 1
        self.transformer = AutoModelForSequenceClassification.from_pretrained(model_type_or_dir, config=self.config)
        self.fp16=fp16
        self.bf16=bf16
        self.dtype = torch.bfloat16 if self.bf16 else torch.float16
        self.force_nofp = force_nofp

    def forward(self, **kwargs):
        if self.force_nofp or (not self.fp16 and not self.bf16):
            context = NullContextManager()
        else:
            context = torch.cuda.amp.autocast(dtype=self.dtype)
        with context:
            # about position embeddings: "they are an optional parameter. If no position IDs are passed to the model,
            # they are automatically created as absolute positional embeddings."
            # see: https://github.com/huggingface/transformers/issues/2287
            return self.transformer(**kwargs)

class RankT5Encoder(torch.nn.Module):
    def __init__(self, model_type_or_dir,fp16=False, bf16=True, force_nofp=False):
        """
        model_type_or_dir is either the name of a pre-trained model (e.g. bert-base-uncased), or the path to
        directory containing model weights, vocab etc.
        """
        super().__init__()
        if "mt5" in model_type_or_dir or "mt0" in model_type_or_dir or "pygaggle" in model_type_or_dir:
            self.model = MT5EncoderModel.from_pretrained(model_type_or_dir)
        else:
            self.model = T5EncoderModel.from_pretrained(model_type_or_dir)
        self.config = self.model.config
        self.first_transform = torch.nn.Linear(self.config.d_model, self.config.d_model)
        self.layer_norm = torch.nn.LayerNorm(self.config.d_model, eps=1e-12)
        self.linear = torch.nn.Linear(self.config.d_model,1)
        self.fp16=fp16
        self.bf16=bf16
        self.dtype = torch.bfloat16 if self.bf16 else torch.float16
        self.force_nofp = force_nofp

    def forward(self, **kwargs):
        if self.force_nofp or (not self.fp16 and not self.bf16):
            context = NullContextManager()
        else:
            context = torch.cuda.amp.autocast(dtype=self.dtype)
        with context:
            result = self.model(**kwargs).last_hidden_state[:,0,:]
            first_transformed = self.first_transform(result)
            layer_normed = self.layer_norm(first_transformed)
            logits = self.linear(layer_normed)
            return SequenceClassifierOutput(
                logits=logits
            )

class RankT5EncoderFix(torch.nn.Module):
    def __init__(self, model_type_or_dir, model_type_or_dir_q=None,fp16=False, bf16=True, force_nofp=False):
        """
        model_type_or_dir is either the name of a pre-trained model (e.g. bert-base-uncased), or the path to
        directory containing model weights, vocab etc.
        """
        assert model_type_or_dir_q is None
        super().__init__()
        if "mt5" in model_type_or_dir or "mt0" in model_type_or_dir or "pygaggle" in model_type_or_dir:
            self.model = MT5EncoderModel.from_pretrained(model_type_or_dir)
        else:
            self.model = T5EncoderModel.from_pretrained(model_type_or_dir)
        self.config = self.model.config
        self.linear = torch.nn.Linear(self.config.d_model,1)
        self.fp16=fp16
        self.bf16=bf16
        self.dtype = torch.bfloat16 if self.bf16 else torch.float16
        self.force_nofp = force_nofp

    def forward(self, **kwargs):
        if self.force_nofp or (not self.fp16 and not self.bf16):
            context = NullContextManager()
        else:
            context = torch.cuda.amp.autocast(dtype=self.dtype)
        with context:
            result = self.model(**kwargs).last_hidden_state[:,0,:]
            logits = self.linear(result)
            return SequenceClassifierOutput(
                logits=logits
            )
