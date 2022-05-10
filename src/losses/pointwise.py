import torch


class BCEWithLogitsLoss:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def __call__(self, out_d):
        pos_scores, neg_scores = out_d["pos_score"], out_d["neg_score"]
        p = pos_scores.squeeze()
        n = neg_scores.squeeze()
        labels = torch.cat([torch.ones(p.shape[0]), torch.zeros(n.shape[0])]).to(self.device)
        return self.loss(torch.cat([p, n]), labels)
