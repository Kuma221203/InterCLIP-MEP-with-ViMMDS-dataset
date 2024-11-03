import logging
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmsd.dataset import MMSDModelInput
from mmsd.model import MMSDOutput

logger = logging.getLogger("mmsd.predictor")

class MemoEnhancedPredictor(nn.Module):

    def __init__(
        self,
        model: Callable[..., MMSDOutput],
        use_memo: bool = True,
        memo_size: int = 512,
        embed_size: int = 512,
    ) -> None:
        super().__init__()
        self.model = model
        self.use_memo = use_memo
        if use_memo:
            self.memo_size = memo_size
            self.register_buffer(
                "entropy", torch.zeros(4, memo_size, device=model.device)
            )
            self.register_buffer(
                "embed_memo_text",
                torch.zeros(4, memo_size, embed_size, device=model.device),
            )
            self.register_buffer(
                "embed_memo_vision",
                torch.zeros(4, memo_size, embed_size, device=model.device),
            )
            self.register_buffer(
                "entropy_memo_ptr", torch.zeros(4, dtype=torch.long, device=model.device)
            )

    def sorted_merge_tensors(
        self,
        selected_entropy : Tensor,
        label : int,
        limit : int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        ptr_memo = self.entropy_memo_ptr[label]

        device = selected_entropy.device

        tensor1 = self.entropy[label, :ptr_memo].to(device)
        tensor2 = selected_entropy

        indices_tensor1 = torch.arange(ptr_memo, device=device)
        indices_tensor2 = torch.arange(len(tensor2), device=device)

        merged_tensor = torch.cat((tensor1, tensor2))
        merged_indices = torch.cat((indices_tensor1, indices_tensor2 + ptr_memo))

        sorted_tensor, sorted_indices = torch.sort(merged_tensor, descending = False)
        sorted_indices = sorted_indices.to(device)
        original_indices = merged_indices[sorted_indices]

        limited_sorted_tensor = sorted_tensor[:limit]
        limited_indices = original_indices[:limit]

        tensor_source = limited_indices >= len(tensor1)
        final_indices = limited_indices.clone()
        final_indices[tensor_source] -= len(tensor1)

        need_to_remain_tensor1_idx = final_indices[~ tensor_source]
        need_to_remain_tensor2_idx = final_indices[tensor_source]

        self.entropy_memo_ptr[label] = limit

        return limited_sorted_tensor, need_to_remain_tensor1_idx, need_to_remain_tensor2_idx

    def fill_memo(
        self,
        selected_entropy: Tensor,
        selected_emebds_text: Tensor,
        selected_emebds_vision: Tensor,
        label : int,
    ) -> Tensor:

        end_idx = self.entropy_memo_ptr[label].item() + len(selected_entropy)
        fill_size = min(self.memo_size, end_idx)
        entropy_availabel, remain_memo_idx, remain_selected_idx = self.sorted_merge_tensors(selected_entropy, label, fill_size)

        embeds_availble_text = torch.cat((self.embed_memo_text[label, remain_memo_idx], selected_emebds_text[remain_selected_idx]))
        self.embed_memo_text[label, :fill_size] = embeds_availble_text

        embeds_availble_vision = torch.cat((self.embed_memo_vision[label, remain_memo_idx], selected_emebds_vision[remain_selected_idx]))
        self.embed_memo_vision[label, :fill_size] = embeds_availble_vision

        self.entropy[label, :fill_size] = entropy_availabel

    def write(
        self,
        model_out: MMSDOutput,
        pseudo_y: Tensor,
        entropy: Tensor,
    ) -> None:
        for label in range(4):
            selected_embeds_text = model_out.text_fused_embeds[pseudo_y == label]
            selected_embeds_vision = model_out.vision_fused_embeds[pseudo_y == label]
            selected_entropy = entropy[pseudo_y == label]
            self.fill_memo(selected_entropy, selected_embeds_text, selected_embeds_vision, label)

    def forward(self, batch: MMSDModelInput) -> tuple[Tensor, Tensor, Tensor]:
        output = self.model(**batch, return_loss=False)

        logits = output.logits
        pred = F.softmax(logits, dim=-1)

        if not self.use_memo:
            return pred

        log_pred = F.log_softmax(logits, dim=-1)

        entropy = -torch.sum(pred * log_pred, dim=-1)
        pseudo_y = torch.argmax(pred, dim=-1)
        self.write(output, pseudo_y, entropy)

        cosin_text = torch.einsum("bd,cmd->bmc", output.text_fused_embeds, self.embed_memo_text)
        cosin_vision = torch.einsum("bd,cmd->bmc", output.vision_fused_embeds, self.embed_memo_vision)

        cosin_text = cosin_text.sum(dim=1)
        cosin_vision = cosin_vision.sum(dim=1)

        cosin_text_combine = torch.stack((cosin_text[:,0] + cosin_text[:,2], cosin_text[:,1] + cosin_text[:,3]), dim = 1)
        cosin_vision_combine = torch.stack((cosin_text[:,0] + cosin_text[:,1], cosin_text[:,2] + cosin_text[:,3]), dim = 1)

        memo_text_pred = F.softmax(cosin_text_combine, dim=-1)
        memo_vision_pred = F.softmax(cosin_vision_combine, dim=-1)

        memo_pred = torch.cat(
                                (
                                    memo_text_pred*memo_vision_pred[:, 0].view(-1, 1),
                                    memo_text_pred*memo_vision_pred[:, 1].view(-1, 1)
                                ), dim = 1
                            )

        return memo_pred, pred, entropy
        # return pred
