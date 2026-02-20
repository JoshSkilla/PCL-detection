from transformers import Trainer
import torch
import torch.nn as nn

class WeightedBCETrainer(Trainer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.float()  # Ensure float
            inputs["labels"] = labels
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        # Weighted BCE loss (neg/pos multiplier with coef is passed in)
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss