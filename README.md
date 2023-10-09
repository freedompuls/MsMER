<div align="center">    
 
# MsMER: A Multi-Scale Feature for Transformer-based Handwritten Mathematical Expression Recognition    


</div>
 
```
# install project   
cd MsMER
conda create -y -n msmer python=3.7
conda activate msmer
conda install --yes -c pytorch pytorch=1.7.0 torchvision cudatoolkit=<your-cuda-version>
pip install -e .   
 ```   
 Next, navigate to any file and run it. It may take ''20-25" hours to coverage on **1** 3080Ti gpus.
 ```bash
# module folder
cd MsMER

python train.py --config config.yaml  
```

For single gpu user, you may change the `config.yaml` file to
```yaml
gpus: 1


![image](https://github.com/freedompuls/MsMER/blob/main/image.png)


