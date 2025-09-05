# smart-turn-rsCLI (quickest):

Command: python -m transformers.onnx --model pipecat-ai/smart-turn-v2 --feature audio-classification --opset 17 --trust-remote-code onnx/smart-turn-v2
Notes:
--trust-remote-code is important if the model repo defines a custom class (likely here).
Output goes to onnx/smart-turn-v2/model.onnx.
Add --use_external_format for very large models.
Programmatic (if the CLI balks on the custom head):

Load your custom class (as in models/test.py) and export via the ONNX utilities:
Create a small wrapper that returns a single tensor (the logits/prob).
Use transformers.onnx’s OnnxConfig (custom subclass) or directly call export with a minimal generate_dummy_inputs for input_values and attention_mask.
This avoids AutoModel mappings that don’t recognize your custom architecture.
Tip: If you prefer Optimum’s wrappers, optimum-cli export onnx -m pipecat-ai/smart-turn-v2 --task audio-classification --trust-remote-code onnx/smart-turn-v2 works similarly.