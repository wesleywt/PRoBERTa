import subprocess
import os

def train():
    os.makedirs('ppi', exist_ok=True)
    subprocess.run(['./pRoBERTa_finetune_ppi.sh',
                    './ppi 1 ./ppi_finetune ./split_binarized/ 768 5 12500 312 0.0025 32 64 2 3 ./trained/current_best_checkpoint/checkpoint_best.pt yes True'])


if __name__ == '__main__':
    train()
