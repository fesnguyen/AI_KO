## Download, extract micromamba
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

## Config path
./bin/micromamba shell init -s bash -r ~/micromamba

## Install torch, cuda
micromamba install python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge

## Install huggingface
micromamba install -y     pytorch torchvision torchaudio pytorch-cuda=12.4     -c pytorch     -c nvidia     -c conda-forge

## Install mlflow
micromamba install -c conda-forge mlflow

## LightGbm CPU still very fast, no need for GPU version which is complex for isntallation
micromamba install -c conda-forge lightgbm

# Start mlflow server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
  
  

findmnt -no SOURCE,UUID -T /swap.img :
/dev/nvme0n1p4 c525251b-c3ed-410e-aaf5-7b032e6e2655

sudo filefrag -v /swap.img | awk '{if($1=="0:"){print $4}}' | sed 's/\..*//'  :
75751424

## Hugging face
### Tokens (read):
llm_download : On Drive


