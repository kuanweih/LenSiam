# LenSiam
**LenSiam** is the self-supervised learning architecture of SimSiam plus a novel domain-specific augmentation for strong gravitational lens images. 


To create a conda env, run:  
```
. create_conda_env.sh
```


To train Limsiam models, run  
```
python  main.py  --data_dir  path/to/your/dataset  -c  configs/train.yaml
```


To calculate UMAP embeddings, run
```
python  calc_umap.py  -c  configs/umap.yaml
```

The [notebook](https://github.com/kuanweih/LenSiam/blob/main/notebooks/NeurIPs_umap_plots.ipynb) contains the code we used to make graphs in our papers in NeurIPS 2023 workshops. 


Note: One of the datasets that are used for UMAP calculation is the HST real images. The dataset can be created by running the code in the repo of [kuanweih/lensed_quasar_database_scraper](https://github.com/kuanweih/lensed_quasar_database_scraper).
