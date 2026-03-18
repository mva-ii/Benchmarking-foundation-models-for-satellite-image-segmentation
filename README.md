### Training on Ruche

Use slurm scripts under `slurm/` to train on Ruche. For example, to train the
AlphaEarth model, run:

More infos on the slurm partitions:

<https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/07_slurm_partitions_description/>

Training on gpu_test:

1. Install uv in your home directory (if not already done)
   <https://docs.astral.sh/uv/getting-started/installation/>:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh

   ```

2. Git clone the repository in your home directory:

   ```bash
   cd $HOME && git clone https://github.com/mva-ii/Benchmarking-foundation-models-for-satellite-image-segmentation && cd Benchmarking-foundation-models-for-satellite-image-segmentation
   ```

3. Install wandb with uv and login to your wandb account:

   ```bash
   uv tool install wandb
   ```

   ```bash
   wb login
   ```

4. Sync venv:

   ```bash
   uv sync
   ```

5. Activate venv:

   ```bash
   source .venv/bin/activate
   ```

6. Run the slurm script:

   ```bash
   sbatch --export=ALL,ENVIRONMENT_ROOT=/gpfs/users/sassis/Benchmarking-foundation-models-for-satellite-image-segmentation/.venv,CONFIG_FILE=/gpfs/users/sassis/Benchmarking-foundation-models-for-satellite-image-segmentation/configurations/fitting/alphaearth_ruche.yaml,PASTIS_R_ROOT=/gpfs/workdir/sassis/data/PASTIS-R,EMBEDDING_ROOT=/gpfs/workdir/sassis/data/AE_EMBEDDING slurm/fit.sh
   ```

   ```bash
   sbatch --export=ALL,ENVIRONMENT_ROOT=/gpfs/users/sassis/Benchmarking-foundation-models-for-satellite-image-segmentation/.venv,CONFIG_FILE=/gpfs/users/sassis/Benchmarking-foundation-models-for-satellite-image-segmentation/configurations/fitting/tessera_ruche.yaml,PASTIS_R_ROOT=/gpfs/workdir/sassis/data/PASTIS-R,EMBEDDING_ROOT=/gpfs/workdir/sassis/data/TESSERA_EMBEDDING slurm/fit.sh
   ```

   ```batch
   sbatch --export=ALL,ENVIRONMENT_ROOT=/gpfs/users/sassis/Benchmarking-foundation-models-for-satellite-image-segmentation/.venv,CONFIG_FILE=/gpfs/users/sassis/Benchmarking-foundation-models-for-satellite-image-segmentation/configurations/fitting/alise_ruche.yaml,PASTIS_R_ROOT=/gpfs/workdir/sassis/data/PASTIS-R,EMBEDDING_ROOT=/gpfs/workdir/sassis/data/ALISE_EMB slurm/fita100.sh
   ```

7. Test it by doing the same command, but with `slurm/test.sh` instead of
   `slurm/fit.sh` and the testing config file instead of the fitting one.
