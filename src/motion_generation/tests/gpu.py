import os
# Forza l'uso della GPU 1 (quella con 40GB+ liberi)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
print(f"ID GPU rilevata da PyTorch: {torch.cuda.current_device()}") 
# Dovrebbe stampare 0, perché per lo script la GPU 1 fisica è diventata la 0 logica.