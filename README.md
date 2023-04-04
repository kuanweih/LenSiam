# simsiam_vit_lensing
SimSiam with ViT encoder for lensing task


Create conda env:  ```. conda_env_setup.sh```


To train Simsiam models, run  
```
python  main.py  --data_dir  path/to/your/dataset  -c  configs/dev.yaml
```


To calculate UMAP embeddings, run
```
python  calc_umap.py  -c  configs/dev_umap.yaml
```