import torch
import matplotlib.pyplot as plt
import numpy as np
from pats.utils import Skeleton2D as SkeletonPATS
from matplotlib.animation import FuncAnimation
from typing import List, Optional, Union
from pathlib import Path

@torch.no_grad()
def render_video(
    original_tokens: torch.Tensor, 
    generated_tokens: torch.Tensor, 
    tokenizer, 
    text: Optional[List[str]] = None, 
    interval: int = 67, 
    smooth_factor: float = 0.5, 
    output_filename: Optional[Union[str, Path]] = None
):
    # 1. Preparazione Dati
    original_pose = tokenizer.dequantize(original_tokens).reshape(-1, 52, 2).cpu().numpy()
    generated_pose = tokenizer.dequantize(generated_tokens).reshape(-1, 52, 2).cpu().numpy()
    parents = SkeletonPATS.parents()

    # Smoothing
    def apply_smooth(p, factor):
        s = np.copy(p)
        for t in range(1, len(s)):
            s[t] = (1 - factor) * p[t] + factor * s[t-1]
        return s

    smoothed_orig = apply_smooth(original_pose, smooth_factor)
    smoothed_gen = apply_smooth(generated_pose, smooth_factor)

    # 2. Setup Figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
    # Aumentiamo il margine inferiore per fare spazio al testo
    plt.subplots_adjust(bottom=0.15) 

    for ax, title in zip([ax1, ax2], ['Original', 'Generated']):
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(1.8, -1.8) # Invertito per orientamento corretto
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Elementi grafici
    lines1 = [ax1.plot([], [], 'k-', lw=2)[0] for _ in range(len(parents))]
    scatter1 = ax1.scatter([], [], c='blue', s=20)
    lines2 = [ax2.plot([], [], 'r-', lw=2)[0] for _ in range(len(parents))]
    scatter2 = ax2.scatter([], [], c='darkred', s=20)

    # Usiamo un font visibile e una posizione sicura
    text_obj = fig.text(0.5, 0.2, '', 
                        ha='center', va='top', 
                        fontsize=12, color='black',
                        fontweight='bold', wrap=True,
                        bbox=dict(facecolor='white', 
                            alpha=0.9, 
                            edgecolor='gray', 
                            boxstyle='round,pad=0.8')
                        )
    

    def update(frame_idx):
        # Aggiornamento Pose 1
        f1 = smoothed_orig[frame_idx]
        for i, p_idx in enumerate(parents):
            if p_idx >= 0:
                lines1[i].set_data([f1[i,0], f1[p_idx,0]], [f1[i,1], f1[p_idx,1]])
        scatter1.set_offsets(f1)

        # Aggiornamento Pose 2
        f2 = smoothed_gen[frame_idx]
        for i, p_idx in enumerate(parents):
            if p_idx >= 0:
                lines2[i].set_data([f2[i,0], f2[p_idx,0]], [f2[i,1], f2[p_idx,1]])
        scatter2.set_offsets(f2)

        # Aggiornamento Testo (mostra le ultime parole per non sovraffollare)
        if text is not None:
            # Prendiamo le parole fino al frame corrente
            words = text[:frame_idx + 1]
            # words = words[-12:]  # ultime 10–12 parole

            filtered = []
            prev = None
            for w in words:
                if w != prev:
                    filtered.append(w)
                prev = w

            display_str = " ".join(filtered)
            text_obj.set_text(display_str)
        
        # Restituiamo tutto
        return lines1 + lines2 + [scatter1, scatter2, text_obj]

    # 3. Creazione Animazione
    # DISATTIVIAMO blit (blit=False) per forzare il ridisegno del testo
    anim = FuncAnimation(
        fig, update, 
        frames=len(smoothed_orig), 
        interval=interval, 
        blit=False 
    )

    # 4. Salvataggio
    if output_filename:
        print(f"Generazione video: {output_filename}...")
        fps = int(1000 / interval)
        anim.save(
            output_filename, 
            writer='ffmpeg', 
            fps=fps, 
            # Parametri critici per la visibilità del testo nei codec video
            extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18']
        )
        print("Salvataggio completato.")
    
    plt.close()
    return anim