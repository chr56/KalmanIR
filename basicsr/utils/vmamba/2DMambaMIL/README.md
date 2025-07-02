# 2DMambaMIL

We prepared the extracted feature in h5 files with the same format from [CLAM library]([/guides/content/editing-an-existing-page](https://github.com/mahmoodlab/CLAM)). 
After preparation, please replace the corresponding h5 directory for the argument `--h5_path`. For CUDA scan, please use the flag `--cuda_pscan`.

You need to first copy the compiled `pscan.so` to `2DMambaMIL/models/pscan_cuda/`. Pay attention to the logs to see if the program actually uses the cuda version.

Sample script to run experiments (we use seed 0,1,2,3,4).

```
cd 2DMambaMIL/2DMambaMIL

CUDA_VISIBLE_DEVICES=0 python main.py --task BRACS --model_type 2DMambaMIL --seed 0 --cuda_pscan --h5_path BRACS_uni/h5_files  
CUDA_VISIBLE_DEVICES=0 python main.py --task BRCA --model_type 2DMambaMIL --seed 0 --cuda_pscan --h5_path BRCA_uni/h5_files  
CUDA_VISIBLE_DEVICES=0 python main.py --task DHMC --model_type 2DMambaMIL --seed 0 --cuda_pscan --h5_path DHMC_uni/h5_files
CUDA_VISIBLE_DEVICES=0 python main.py --task NSCLC --model_type 2DMambaMIL --seed 0 --cuda_pscan --h5_path NSCLC_uni/h5_files 
CUDA_VISIBLE_DEVICES=0 python main.py --task PANDA --model_type 2DMambaMIL --seed 0 --cuda_pscan --h5_path PANDA_uni/h5_files

CUDA_VISIBLE_DEVICES=0 python main.py --task KIRC --survival --model_type 2DMambaMIL --seed 0 --cuda_pscan --h5_path KIRC_uni/h5_files  
CUDA_VISIBLE_DEVICES=0 python main.py --task KIRP --survival --model_type 2DMambaMIL --seed 0 --cuda_pscan --h5_path KIRP_uni/h5_files  
CUDA_VISIBLE_DEVICES=0 python main.py --task LUAD --survival --model_type 2DMambaMIL --seed 0 --cuda_pscan --h5_path LUAD_uni/h5_files 
CUDA_VISIBLE_DEVICES=0 python main.py --task STAD --survival --model_type 2DMambaMIL --seed 0 --cuda_pscan --h5_path STAD_uni/h5_files
CUDA_VISIBLE_DEVICES=0 python main.py --task UCEC --survival --model_type 2DMambaMIL --seed 0 --cuda_pscan --h5_path UCEC_uni/h5_files
``` 
