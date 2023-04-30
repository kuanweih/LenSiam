# simsiam_vit_lensing
SimSiam with ViT encoder for lensing task


Create conda env:  ```. conda_env_setup.sh```


To train Simsiam models, run  
```
python  main.py  --data_dir  path/to/your/dataset  -c  configs/train.yaml
```


To calculate UMAP embeddings, run
```
python  calc_umap.py  -c  configs/umap.yaml
```

Note: One of the datasets that are used for UMAP calculation is the HST real images. The dataset can be created by running the code in the repo of [kuanweih/lensed_quasar_database_scraper](https://github.com/kuanweih/lensed_quasar_database_scraper).