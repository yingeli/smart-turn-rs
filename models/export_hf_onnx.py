import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch import nn

from transformers import (
    AutoConfig,
    AutoProcessor,
    Wav2Vec2Config,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from transformers.onnx import FeaturesManager, export


class _Wav2Vec2ForEndpointing(Wav2Vec2PreTrainedModel):
    """Custom endpointing head used by the HF repo."""

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
        attention_weights = attention_weights + (
            (1.0 - attention_mask.unsqueeze(-1).to(attention_weights.dtype)) * -1e9
        )
        attention_weights = torch.softmax(attention_weights, dim=1)
        return torch.sum(hidden_states * attention_weights, dim=1)

    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
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

        pooled = self.attention_pool(hidden_states, attention_mask)
        logits = self.classifier(pooled)
        probs = torch.sigmoid(logits)
        # Return a plain tensor; we'll wrap below to produce a dict with "logits"
        return probs


class HFEndpointingWrapper(nn.Module):
    """Wraps the custom model to present a transformers.onnx-friendly output."""

    def __init__(self, model: _Wav2Vec2ForEndpointing):
        super().__init__()
        self.model = model
        # Preserve config attribute so transformers.onnx picks the right OnnxConfig
        self.config = model.config

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor):
        out = self.model(input_values=input_values, attention_mask=attention_mask)
        # transformers.onnx expects a mapping with names from OnnxConfig.outputs
        return {"logits": out}


def export_with_transformers_onnx(
    model_id_or_path: str,
    output_dir: Path,
    feature: str = "audio-classification",
    opset: int = 17,
    trust_remote_code: bool = False,
    use_external_format: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preprocessor (processor handles padding + attention_mask for audio)
    processor = AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)

    # Load our custom model class and wrap for ONNX export
    model = _Wav2Vec2ForEndpointing.from_pretrained(model_id_or_path)
    model.eval()
    wrapper = HFEndpointingWrapper(model).eval()

    # Let transformers pick the appropriate ONNX config for Wav2Vec2 + audio-classification
    _, onnx_config_cls = FeaturesManager.check_supported_model_or_raise(wrapper, feature=feature)
    onnx_config = onnx_config_cls(wrapper.config)

    output_path = output_dir / "model.onnx"

    export(
        preprocessor=processor,
        model=wrapper,
        config=onnx_config,
        opset=opset,
        output=output_path,
        use_external_format=use_external_format,
    )

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export pipecat-ai/smart-turn-v2 via transformers.onnx")
    parser.add_argument("--model", default="pipecat-ai/smart-turn-v2", help="HF repo id or local path")
    parser.add_argument("--out-dir", default="onnx/smart-turn-v2", help="Output directory")
    parser.add_argument("--feature", default="audio-classification", help="Feature (default: audio-classification)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code from the model repo")
    parser.add_argument("--use-external-format", action="store_true", help="Export with external data format")
    args = parser.parse_args()

    out = export_with_transformers_onnx(
        model_id_or_path=args.model,
        output_dir=Path(args.out_dir),
        feature=args.feature,
        opset=args.opset,
        trust_remote_code=args.trust_remote_code,
        use_external_format=args.use_external_format,
    )
    print(f"Exported ONNX model to: {out}")


if __name__ == "__main__":
    main()

