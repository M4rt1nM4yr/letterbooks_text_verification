# Text Verification - The Nuremberg Letterbooks: A Comprehensive Multi-Transcriptional Dataset of Early 15th Century Manuscripts for Document Analysis

## Overview
This project provides tools for text verification using the Nuremberg Letterbooks dataset.

## Usage

- Adjust root path in the datamodule configs. Root folder should contain all the image folders and the transcription folders (nuremberg_letterbooks) downloaded from Zenodo.
- Run training: 
	python train.py --config configs/your_config.yaml
- For SLURM clusters:
	sbatch slurm.sh train.py "version=test"


