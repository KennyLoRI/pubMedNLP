from sentence_transformers import SentenceTransformer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO', device=device)
model.max_seq_length = 512

def embed(input: list[str]) -> list[torch.Tensor]:
    embedding = model.encode(input, device=device, convert_to_numpy=False)
    return embedding