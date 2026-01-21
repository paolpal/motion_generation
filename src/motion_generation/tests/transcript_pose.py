import random
from motion_generation.dataset import TranscriptPoseDataset
from motion_quantization.quantization import PoseTokenizer
from motion_quantization.models import SkeletonVQVAE
from motion_generation.tokenizer import Word2VecTokenizer

ROOT_PRJ = "/home/paolo/Projects/Posemi"

model = SkeletonVQVAE.load(f"{ROOT_PRJ}/motion_quantization/scripts/weights/vqvae_trial_8.pt")

pose_tokenizer = PoseTokenizer(model=model, device='cpu')
text_tokenizer = Word2VecTokenizer.load(f"{ROOT_PRJ}/motion_generation/weights/tokenizer/word_tokenizer.json")
dataset = TranscriptPoseDataset(
    speakers=['fallon'],
    split="dev",
    data_root="/home/paolo/Projects/Gesture/pats/data",
    pose_tokenizer=pose_tokenizer,
    text_tokenizer=text_tokenizer,
    cache_path=f"{ROOT_PRJ}/motion_generation/tests/cache/transcript_pose_fallon_dev.pt"
)

print("Dataset length:", len(dataset))

idx = random.randint(0, len(dataset)-1)

text_tokens, motion_tokens = dataset[idx]

print("Text Tokens:", text_tokens)
print("Text Tokens shape:", text_tokens.shape)
# print("Words:", clip['words'])
print("Motion Tokens:", motion_tokens)
print("Motion Tokens shape:", motion_tokens.shape)

