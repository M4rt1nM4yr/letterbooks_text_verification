#!/bin/bash
#SBATCH --partition=a40
#SBATCH --job-name=DiplomaticHTR
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err
#SBATCH --time=24:00:00

export HYDRA_FULL_ERROR=1

cd /home/woody/iwi5/iwi5059h/project/nbb_text_verification
#module load python/3.9-anaconda
source /apps/python/3.9-anaconda/etc/profile.d/conda.sh
conda activate currentPL

export HDF5_USE_FILE_LOCKING=FALSE
export https_proxy="http://proxy.rrze.uni-erlangen.de:80"
# Check if the proxy is working
curl -I https://api.wandb.ai

# Proceed only if the proxy check was successful
if [ $? -eq 0 ]; then
    echo "Proxy is working."
    # Run your script
    echo "Your job is running on" $(hostname)
    DEVICES=${3:-1}
    srun python3 $1 $2 trainer.devices=$DEVICES
else
    echo "Proxy is not working. Exiting."
    exit 1
fi
