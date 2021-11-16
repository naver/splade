#FROM Sentence-BERT: (https://github.com/UKPLab/sentence-transformers/blob/afee883a17ab039120783fd0cffe09ea979233cf/sentence_transformers/losses/MultipleNegativesRankingLoss.py) (https://github.com/UKPLab/sentence-transformers/blob/afee883a17ab039120783fd0cffe09ea979233cf/sentence_transformers/losses/MarginMSELoss.py) (https://github.com/UKPLab/sentence-transformers/blob/afee883a17ab039120783fd0cffe09ea979233cf/sentence_transformers/util.py) with minimal changes.
#Original License APACHE2

import torch
from torch import nn, Tensor
from typing import Iterable, Dict

def pairwise_dot_score(a: Tensor, b: Tensor):
    """
   Computes the pairwise dot-product dot_prod(a[i], b[i])
   :return: Vector with res[i] = dot_prod(a[i], b[i])
   """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    return (a * b).sum(dim=-1)

def dot_score(a: Tensor, b: Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))

class FLOPS:
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __call__(self, batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)

class MarginMSELossSplade(nn.Module):
    """
    Compute the MSE loss between the |sim(Query, Pos) - sim(Query, Neg)| and |gold_sim(Q, Pos) - gold_sim(Query, Neg)|
    By default, sim() is the dot-product
    For more details, please refer to https://arxiv.org/abs/2010.02666
    """
    def __init__(self, model, similarity_fct = pairwise_dot_score, lambda_d=8e-2, lambda_q=1e-1):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use
        """
        super(MarginMSELossSplade, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.loss_fct = nn.MSELoss()
        self.lambda_d = lambda_d
        self.lambda_q = lambda_q
        self.FLOPS = FLOPS()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_query = reps[0]
        embeddings_pos = reps[1]
        embeddings_neg = reps[2]

        scores_pos = self.similarity_fct(embeddings_query, embeddings_pos)
        scores_neg = self.similarity_fct(embeddings_query, embeddings_neg)
        margin_pred = scores_pos - scores_neg

        flops_doc = self.lambda_d*(self.FLOPS(embeddings_pos) + self.FLOPS(embeddings_neg))
        flops_query = self.lambda_q*(self.FLOPS(embeddings_query))

        return self.loss_fct(margin_pred, labels) + flops_doc + flops_query

class MultipleNegativesRankingLossSplade(nn.Module):
    """
        This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2), ..., (a_n, p_n)
        where we assume that (a_i, p_i) are positive pairs and (a_i, p_j) for i!=j negative pairs.
        For each a_i, it uses all other p_j as negative samples, i.e., for a_i, we have 1 positive example (p_i) and
        n-1 negative examples (p_j). It then minimizes the negative log-likehood for softmax normalized scores.
        This loss function works great to train embeddings for retrieval setups where you have positive pairs (e.g. (query, relevant_doc))
        as it will sample in each batch n-1 negative docs randomly. The performance usually increases with increasing batch sizes.
        For more information, see: https://arxiv.org/pdf/1705.00652.pdf
        (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)
        You can also provide one or multiple hard negatives per anchor-positive pair by structering the data like this:
        (a_1, p_1, n_1), (a_2, p_2, n_2)
        Here, n_1 is a hard negative for (a_1, p_1). The loss will use for the pair (a_i, p_i) all p_j (j!=i) and all n_j as negatives.
    """
    def __init__(self, model, scale: float = 1.0, similarity_fct = dot_score, lambda_d=0.0008, lambda_q=0.0006):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(MultipleNegativesRankingLossSplade, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.lambda_d = lambda_d
        self.lambda_q = lambda_q
        self.FLOPS = FLOPS()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])

        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]

        flops_doc = self.lambda_d*(self.FLOPS(embeddings_b))
        flops_query = self.lambda_q*(self.FLOPS(embeddings_a))

        return self.cross_entropy_loss(scores, labels) + flops_doc + flops_query

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__, "lambda_q": self.lambda_q, "lambda_d": self.lambda_d}
