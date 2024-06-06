import torch


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}
        self.emb_name = 'word_embeddings.weight'

    def attack(self, epsilon=1.):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def save_checkpoint(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()

    def load_checkpoint(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]