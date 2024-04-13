import os
import random
import subprocess

try:
    import numpy as np
except:
    print(
        "WARNING: Aparentemente não tem numpy instalado. Usado para estrutura de matriz rápidas. Instale com `pip install numpy`"
    )
    exit(1)

try:
    from pathlib import Path
except:
    print("WARNING: pathlib is not found. Python version might be less than 3.4")


import pim


def string(img, letters):
    print(img.shape)
    orig = pim.convert_to_rgb(img)
    img = pim.invert(img)
    bbxs = pim.find_connected_components_bboxes(img, connectivity=8)
    # Start from leftmost
    bbxs = sorted(bbxs, key=lambda b: b[1])

    letter_sequence = []
    for b in bbxs:
        y, x, y2, x2 = b
        orig = pim.rectangle(orig, (y, x), (y2, x2), (255, 0, 0), 1)
        best_sim = 0
        best_letter = "-"
        for l, limg in letters.items():
            resized = pim.resize(img[y:y2, x:x2], *limg.shape)
            sim = 0
            for j in range(limg.shape[0]):
                for i in range(limg.shape[1]):
                    if resized[j, i] > 0:
                        if limg[j, i] > 0:
                            sim += 1

            pim.write_ppm_file(f"./output/resized{b}.ppm", resized)
            # sim = (limg == resized).flatten().sum() # Outra tentativa
            print(f"{l} -> {sim=}")

            if sim > best_sim:
                best_sim = sim
                best_letter = l
        letter_sequence.append(best_letter)
        print(letter_sequence)

    return "".join(letter_sequence), orig


def main():
    words = ["volta", "tremendo", "nesse"]
    for w in words:
        img = pim.read_ppm_file(f"./assets/word_{w}.ppm")

        letters = {
            l: pim.invert(pim.read_ppm_file(f"./assets/letters/{l}.pbm"))
            for l in [
                "v",
                "o",
                "l",
                "t",
            ]
        }

        found, img = string(img, letters)
        print(f"INFO: Found {found} for {w}")
        pim.write_ppm_file(f"./output/ocr_{w}.ppm", img)


if __name__ == "__main__":
    main()
