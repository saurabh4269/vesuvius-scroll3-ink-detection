"""Patch unetr_segformer.py to use safetensors loading (bypasses torch.load pickle check)."""
import sys

path = (
    "/home/medal/shiwani.mishra/scroll_prize/vesuvius_first_title_prize"
    "/src/phoenix/model/architectures/unetr_segformer.py"
)

with open(path) as f:
    src = f.read()

old = (
    "            self.encoder_2d = SegformerForSemanticSegmentation.from_pretrained(\n"
    "                pretrained_model_name_or_path=segformer_from_pretrained,\n"
    "                num_channels=unetr_out_channels,\n"
    "                ignore_mismatched_sizes=True,\n"
    "                num_labels=1,\n"
    "            )"
)
new = (
    "            self.encoder_2d = SegformerForSemanticSegmentation.from_pretrained(\n"
    "                pretrained_model_name_or_path=segformer_from_pretrained,\n"
    "                num_channels=unetr_out_channels,\n"
    "                ignore_mismatched_sizes=True,\n"
    "                num_labels=1,\n"
    "                use_safetensors=True,\n"
    "            )"
)

if old not in src:
    print("WARNING: patch target not found")
    sys.exit(1)

src = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("unetr_segformer.py patched: use_safetensors=True added")
