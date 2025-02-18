from typing import Optional

import torch


class Wrapper(torch.nn.Module):
    def __init__(self, model: Optional[torch.nn.Module] = None):
        super(Wrapper, self).__init__()
        self.model = model

    def forward(self, x, n_layers: int = 4, **kwargs):
        if self.model is not None:
            return self.model(x, **kwargs)
        else:
            raise ValueError("No model was defined.")

    @property
    def device(self):
        return next(self.parameters()).device


class ViTWrapper(Wrapper):
    def __init__(
        self,
        model: torch.nn.Module,
        head: torch.nn.Module = torch.nn.Identity(),
    ):
        super(ViTWrapper, self).__init__(model=model)
        if type(head) is not torch.nn.Identity:
            self.head = head.mlp
        else:
            self.head = head

    def forward(self, x, n_layers: int = 4, return_all_tokens: bool = False, **kwargs):
        # extract the embeddings from the last N layers and combine
        inter_out = self.model.get_intermediate_layers(x, n_layers)
        if return_all_tokens:
            emb = torch.cat(inter_out, dim=-1)
        else:
            emb = torch.cat([x[:, 0, :] for x in inter_out], dim=-1)
        emb = self.head(emb)
        return emb


class ViTHuggingFaceWrapper(Wrapper):
    def __init__(
        self,
        vit_huggingface_name: str = "WinKawaks/vit-tiny-patch16-224",
        n_layers: Optional[int] = None,
    ):
        super(ViTHuggingFaceWrapper, self).__init__()
        self.n_layers = n_layers
        from transformers import ViTModel

        self.model = ViTModel.from_pretrained(vit_huggingface_name)

    def forward(
        self,
        x: torch.Tensor,
        n_layers: int = 4,
        return_all_tokens: bool = False,
        **kwargs
    ):
        if self.n_layers is not None:
            n_layers = self.n_layers
        # extract the embeddings from the ViT
        if n_layers == 1:
            inter_out = self.model(
                pixel_values=x,
                # makes sure that all input resolutions are allowed
                interpolate_pos_encoding=True,
            )["last_hidden_state"]
            if return_all_tokens:
                return inter_out
            else:
                return inter_out[:, 0, :]
        else:
            hidden_states = self.model(
                pixel_values=x,
                output_hidden_states=True,
                # makes sure that all input resolutions are allowed
                interpolate_pos_encoding=True,
            )["hidden_states"]
            inter_out = hidden_states[-n_layers:]
            if return_all_tokens:
                emb = torch.cat(inter_out, dim=-1)
            else:
                emb = torch.cat([x[:, 0, :] for x in inter_out], dim=-1)
            return emb
