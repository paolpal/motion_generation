import matplotlib.pyplot as plt

from typing import Union
from pathlib import Path
import math
from typing import Union, List
import torch

def save_attention_heatmap(attn, x_labels = None, out_path: Union[str, Path] ="attn_heatmap.png"):
    """
    attn: Tensor (tgt_len, src_len)
    x_labels: list[str]
    """
    attn = attn.cpu().numpy()

    fig, ax = plt.subplots(figsize=(0.6 * attn.shape[1], 0.4 * attn.shape[0]))

    cmap = plt.cm.get_cmap('Blues')
    cmap.set_bad(color='#EEEEEE')

    im = ax.imshow(attn, aspect="auto", interpolation="nearest", cmap=cmap)

    ax.set_xlabel("Text tokens")
    ax.set_ylabel("Gesture frames")

    if x_labels is not None:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

    ax.set_yticks(range(attn.shape[0]))

    # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if isinstance(out_path, str):
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=75)
    plt.close()


def save_attention_subplots(attns: List[torch.Tensor], x_labels=None, out_path: Union[str, Path]="all_heads.png"):
    num_heads = len(attns)
    # Calcoliamo una griglia quadrata o rettangolare ottimale
    cols = min(2, num_heads) 
    rows = math.ceil(num_heads / cols)

    # Regola la dimensione della figura in base al numero di heads
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    
    cmap = plt.cm.get_cmap('Blues')
    cmap.set_bad(color='#EEEEEE')

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < num_heads:
            attn = attns[i].detach().cpu().numpy()
            im = ax.imshow(attn, aspect="auto", interpolation="nearest", cmap=cmap)
            ax.set_title(f"Head {i+1}")
            
            # Etichette solo se passate e solo per gli assi necessari
            if x_labels is not None:
                ax.set_xticks(range(len(x_labels)))
                ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        else:
            # Nascondi i subplot vuoti se num_heads non è un multiplo di cols
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()