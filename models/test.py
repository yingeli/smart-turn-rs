from transformers import Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Processor
import soundfile as sf
import torch
from torch import nn
import torch.nn.functional as F
import time

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
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()

        for module in self.pool_attention:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()

    def attention_pool(self, hidden_states, attention_mask):
        # Calculate attention weights
        attention_weights = self.pool_attention(hidden_states)

        if attention_mask is None:
            raise ValueError("attention_mask must be provided for attention pooling")

        attention_weights = attention_weights + (
            (1.0 - attention_mask.unsqueeze(-1).to(attention_weights.dtype)) * -1e9
        )

        attention_weights = F.softmax(attention_weights, dim=1)

        # Apply attention to hidden states
        weighted_sum = torch.sum(hidden_states * attention_weights, dim=1)

        return weighted_sum

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]

        # Create transformer padding mask
        if attention_mask is not None:
            input_length = attention_mask.size(1)
            hidden_length = hidden_states.size(1)
            ratio = input_length / hidden_length
            indices = (torch.arange(hidden_length, device=attention_mask.device) * ratio).long()
            attention_mask = attention_mask[:, indices]
            attention_mask = attention_mask.bool()
        else:
            attention_mask = None

        pooled = self.attention_pool(hidden_states, attention_mask)

        logits = self.classifier(pooled)

        if torch.isnan(logits).any():
            raise ValueError("NaN values detected in logits")

        if labels is not None:
            # Calculate positive sample weight based on batch statistics
            pos_weight = ((labels == 0).sum() / (labels == 1).sum()).clamp(min=0.1, max=10.0)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1), labels.view(-1))

            # Add L2 regularization for classifier layers
            l2_lambda = 0.01
            l2_reg = torch.tensor(0.0, device=logits.device)
            for param in self.classifier.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg

            probs = torch.sigmoid(logits.detach())
            return {"loss": loss, "logits": probs}

        probs = torch.sigmoid(logits)
        return {"logits": probs}

model = _Wav2Vec2ForEndpointing.from_pretrained("pipecat-ai/smart-turn-v2", device_map="cuda")
processor = Wav2Vec2Processor.from_pretrained("pipecat-ai/smart-turn-v2")

speech, sr = sf.read("../smart-turn-rs/models/audio1.wav")
if sr != 16_000:
    raise ValueError("Resample to 16 kHz")

inputs = processor(
    speech,
    sampling_rate=16000,
    padding="max_length",
    truncation=True,
    max_length=16000 * 16,  # 16 seconds at 16kHz
    return_attention_mask=True,
    return_tensors="pt",
)

inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

    start = time.time()
    outputs = model(**inputs)
    duration = time.time() - start
    print(f"Inference completed in {duration:.3f} seconds")

    # The model returns sigmoid probabilities directly in the logits field
    probability = outputs["logits"][0].item()

    # Make prediction (1 for Complete, 0 for Incomplete)
    prediction = 1 if probability > 0.5 else 0

print(f"Completed turn? {prediction}  Prob: {probability:.3f}")
# label == 'complete' → user has finished speaking