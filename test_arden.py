import sys
import torch
sys.path.insert(0, '/opt/arden')
from core.config import ArdenConfig
from core.model import ArdenModel
from core.tokenizer import ArdenTokenizer
from pathlib import Path
cfg       = ArdenConfig()
vocab_dir = Path('/opt/arden/data/vocab')
ckpt_path = Path('/opt/arden/checkpoints/best_model.pt')

print("Cargando tokenizer...")
tokenizer = ArdenTokenizer.load(vocab_dir)

print("Cargando modelo...")
model = ArdenModel(cfg.model)
ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model_state'])
model.eval()

def generate(prompt, max_new_tokens=50, temperature=0.8):
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    x   = torch.tensor([ids], dtype=torch.long)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out    = model(input_ids=x)
            logits = out['lm_logits'][:, -1, :]
            # Repetition penalty
            for tok in set(x[0].tolist()):
                if logits[0, tok] > 0:
                    logits[0, tok] /= 1.3
                else:
                    logits[0, tok] *= 1.3
            logits = logits / temperature
            k = 40
            kth = torch.topk(logits, k)[0][:, -1:]
            logits = logits.masked_fill(logits < kth, float('-inf'))
            probs  = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            x = torch.cat([x, next_id], dim=1)
            if next_id.item() == tokenizer.eos_id:
                break
    
    return tokenizer.decode(x[0].tolist())

print("\n--- PRUEBA ARDEN 0.2 ---\n")
prompts = [
    "The history of the world is",     # EN (77% del corpus)
    "La historia del mundo es",        # ES (5% — el más débil)
    "L'histoire du monde est",         # FR (18%)
    "A história do mundo é",           # PT (Wikipedia)
]

for p in prompts:
    print(f"Prompt : {p}")
    print(f"Arden  : {generate(p)}")
    print()
