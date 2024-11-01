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
 Next, navigate to any file and run it. It may take ''20-25" hours to coverage on **1** 3080Ti gpu.
 ```bash
# module folder
cd MsMER

python train.py --config config.yaml  
```

For single gpu user, you may change the `config.yaml` file to，
```yaml
gpus: 1
#  accelerator: ddp

```

data CROHME https://www.cs.rit.edu/~crohme2019/task.html
```
our aug data  36260 images
链接：https://pan.baidu.com/s/1ZDSN0skrSmcdqBxebpAX7w 
提取码：x9ng

```

![image](https://github.com/freedompuls/MsMER/blob/main/image.png)

![image](https://github.com/freedompuls/MsMER/blob/main/image1.PNG)

![image](https://github.com/freedompuls/MsMER/blob/main/image2.PNG)
