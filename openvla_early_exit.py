import torch
import torch.nn as nn
from openvla.models.prismatic import PrismaticForVisionAndLanguageGeneration
from transformers import BitsAndBytesConfig

class PrismaticEarlyExit(PrismaticForVisionAndLanguageGeneration):
    def __init__(self, config, exit_layers=[4, 8], exit_threshold=0.95):
        super().__init__(config)
        # Define layers for early exit
        self.exit_layers = exit_layers
        self.exit_threshold = exit_threshold
        # Add action prediction heads for each exit layer (match main head structure)
        self.exit_heads = nn.ModuleDict()
        for layer_idx in exit_layers:
            self.exit_heads[str(layer_idx)] = nn.Linear(config.hidden_size, 7)  # 7D action vector
        # Confidence calculation (lower variance = higher confidence)
        self.confidence_fn = lambda x: 1 - torch.var(x, dim=-1).mean()

    def forward(
        self,
        pixel_values=None,
        text_input_ids=None,
        attention_mask=None,
        return_dict=True,
        **kwargs
    ):
        # Visual feature extraction (reuse original logic)
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        vision_embeds = vision_outputs.last_hidden_state

        # Text feature extraction (reuse original logic)
        text_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        # Fusion + Transformer layers with early exit check
        fused_embeds = self.fusion_layer(vision_embeds, text_embeds)
        transformer_outputs = fused_embeds
        exit_pred = None
        exit_layer_used = None

        for layer_idx, layer in enumerate(self.transformer.layers):
            transformer_outputs = layer(transformer_outputs, attention_mask=attention_mask)
            
            # Early exit check: validate confidence if current layer is an exit layer
            if layer_idx in self.exit_layers:
                exit_logits = self.exit_heads[str(layer_idx)](transformer_outputs[:, 0, :])
                confidence = self.confidence_fn(exit_logits)
                
                if confidence >= self.exit_threshold:
                    exit_pred = exit_logits
                    exit_layer_used = layer_idx
                    break  # Trigger early exit, stop layer computation

        # Use main prediction head if no early exit triggered
        if exit_pred is None:
            final_embeds = self.transformer.pooler(transformer_outputs)
            exit_pred = self.action_head(final_embeds)
            exit_layer_used = len(self.transformer.layers)

        if return_dict:
            return {
                "action_pred": exit_pred,
                "exit_layer": exit_layer_used,
                "confidence": self.confidence_fn(exit_pred)
            }
        return (exit_pred, exit_layer_used)

    # Adapt to original predict_action interface
    def predict_action(self, **kwargs):
        outputs = self.forward(**kwargs)
        return outputs["action_pred"]