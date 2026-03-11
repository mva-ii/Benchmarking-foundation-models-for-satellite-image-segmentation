### Training on Ruche

Use slurm scripts under `slurm/` to train on Ruche. For example, to train the
AlphaEarth model, run:

More infos on the slurm partitions:

<https://mesocentre.pages.centralesupelec.fr/user_doc/ruche/07_slurm_partitions_description/>

Training on gpu_test:

```bash
sbatch --export=ALL,ENVIRONMENT_ROOT=/gpfs/users/sassis/Benchmarking-foundation-models-for-satellite-image-segmentation/.venv,CONFIG_FILE=/gpfs/users/sassis/Benchmarking-foundation-models-for-satellite-image-segmentation/configurations/fitting/alphaearth_ruche.yaml,PASTIS_R_ROOT=/gpfs/workdir/sassis/data/PASTIS-R,EMBEDDING_ROOT=/gpfs/workdir/sassis/data/AE_EMBEDDING slurm/fit.sh
```
