import sys
import torch
sys.path.insert(0, '/opt/arden')
from core.config_05b import get_05b_config
from core.model import ArdenModel
from core.tokenizer import ArdenTokenizer
from pathlib import Path

cfg       = get_05b_config()
vocab_dir = Path('/opt/arden/data/vocab')
ckpt_path = Path('/opt/arden/checkpoints_05b/best_model.pt')

print("Cargando tokenizer...")
tokenizer = ArdenTokenizer.load(vocab_dir)

print("Cargando modelo...")
model = ArdenModel(cfg.model)
ckpt  = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(ckpt['model_state'])
model.eval()

def generate(prompt, max_new_tokens=50, temperature=0.8):
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    x   = torch.tensor([ids], dtype=torch.long)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out    = model(input_ids=x)
            logits = out['lm_logits'][:, -1, :] / temperature
            probs  = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            x = torch.cat([x, next_id], dim=1)
            if next_id.item() == tokenizer.eos_id:
                break
    
    return tokenizer.decode(x[0].tolist())

print("\n--- PRUEBA ARDEN 0.2 ---\n")
prompts = [
    "The European Parliament",
    "El Parlamento Europeo",
    "Le Parlement européen",
]

for p in prompts:
    print(f"Prompt : {p}")
    print(f"Arden  : {generate(p)}")
    print()
