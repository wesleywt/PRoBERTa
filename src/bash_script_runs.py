import subprocess


def binary():
    subprocess.run(['./binarize_input.sh'], shell=True)


def train():
    subprocess.run(['bash',
                    'ppi 1 ppi_finetune split_binarized/ 768 5 12500 312 0.0025 32 64 2 3 trained/ppi.DIM_768.LAYERS_5.UPDATES_12500.WARMUP_312.LR_0.0025.BATCH_2048.PATIENCE_3/checkpoints/checkpoint_best.pt no True'])
