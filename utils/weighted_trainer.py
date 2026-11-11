import torch
from torch import nn
from transformers import Trainer


class WeightedTrainer(Trainer):
    """
    Customize the Trainer with weighted cross-entropy loss.
    docs: https://huggingface.co/docs/transformers/v4.35.2/main_classes/trainer
    """

    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        # inherit from Trainer class
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device), ignore_index=-100)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
