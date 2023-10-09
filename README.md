<div align="center">    
 
# MsMER: A Multi-Scale Feature for Transformer-based Handwritten Mathematical Expression Recognition    


</div>
 
```
# install project   
cd MsMER
conda create -y -n bttr python=3.7
conda activate bttr
conda install --yes -c pytorch pytorch=1.7.0 torchvision cudatoolkit=<your-cuda-version>
pip install -e .   
 ```   
 Next, navigate to any file and run it. It may take 20 hours to coverage on **1** 3080Ti gpus.
 ```bash
# module folder
cd MsMER

python train.py --config config.yaml  
```

For single gpu user, you may change the `config.yaml` file to
```yaml
gpus: 1

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from bttr.datamodule import CROHMEDatamodule
from bttr import LitBTTR
from pytorch_lightning import Trainer

# model
model = LitMSMER()

# data
dm = CROHMEDatamodule(test_year=test_year)

# train
trainer = Trainer()
trainer.fit(model, datamodule=dm)


![image]([https://github.com/freedompuls/MsMER/blob/main/image.png](https://github.com/freedompuls/MsMER/blob/main/image.png)https://github.com/freedompuls/MsMER/blob/main/image.png)


