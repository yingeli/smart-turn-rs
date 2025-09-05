import argparse
import os
from typing import Optional

import torch
from torch import nn

from transformers import (
    Wav2Vec2Config,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


class _Wav2Vec2ForEndpointing(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)

        self.pool_attention = nn.Sequential(
            nn.Linear(config.hidden_size, 256), nn.Tanh(), nn.Linear(256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for module in self.pool_attention:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def attention_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        attention_weights = self.pool_attention(hidden_states)

        if attention_mask is None:
            raise ValueError("attention_mask must be provided for attention pooling")

        attention_weights = attention_weights + (
            (1.0 - attention_mask.unsqueeze(-1).to(attention_weights.dtype)) * -1e9
        )

        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted_sum = torch.sum(hidden_states * attention_weights, dim=1)
        return weighted_sum

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]

        if attention_mask is not None:
            input_length = attention_mask.size(1)
            hidden_length = hidden_states.size(1)
            ratio = input_length / hidden_length
            indices = (
                torch.arange(hidden_length, device=attention_mask.device) * ratio
            ).long()
            attention_mask = attention_mask[:, indices]
            attention_mask = attention_mask.bool()
        else:
            attention_mask = None

        pooled = self.attention_pool(hidden_states, attention_mask)
        logits = self.classifier(pooled)
        probs = torch.sigmoid(logits)
        return probs


class LogitsOnlyWrapper(nn.Module):
    """
    torch.onnx.export works best with tensor outputs.
    Wrap the model to return a single tensor.
    """

    def __init__(self, model: _Wav2Vec2ForEndpointing):
        super().__init__()
        self.model = model

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor):
        return self.model(input_values=input_values, attention_mask=attention_mask)


def export(
    model_id_or_path: str,
    output_path: str,
    opset: int = 17,
    seq_len: int = 16000 * 16,
    device: str = "cpu",
):
    model = _Wav2Vec2ForEndpointing.from_pretrained(model_id_or_path)
    model.eval()
    model.to(device)

    wrapper = LogitsOnlyWrapper(model)
    wrapper.eval().to(device)

    dummy_input = torch.zeros(1, seq_len, dtype=torch.float32, device=device)
    dummy_mask = torch.ones(1, seq_len, dtype=torch.int64, device=device)

    dynamic_axes = {
        "input_values": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch"},
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy_input, dummy_mask),
        output_path,
        input_names=["input_values", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export pipecat-ai/smart-turn-v2 to ONNX"
    )
    parser.add_argument(
        "--model",
        default="pipecat-ai/smart-turn-v2",
        help="HF repo id or local model directory",
    )
    parser.add_argument(
        "--output",
        default="models/smart-turn-v2.onnx",
        help="Path to write ONNX model",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=16000 * 16,
        help="Dummy input sequence length used for export",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Export device"
    )
    args = parser.parse_args()

    export(
        model_id_or_path=args.model,
        output_path=args.output,
        opset=args.opset,
        seq_len=args.seq_len,
        device=args.device,
    )

    print(f"Exported ONNX model to: {args.output}")


if __name__ == "__main__":
    main()

