from msmer.lit_msmer import LitMSMER
from PIL import Image
from torchvision.transforms import ToTensor
from msmer.datamodule import vocab
import os


ckpt = '/checkpoints/epoch=175-step=253439-val_ExpRate=0.5446.ckpt'

img_path = '18_em_1.bmp'
model = LitMSMER.load_from_checkpoint(ckpt)

# print(model)
img = Image.open(img_path)
print(img)
img = ToTensor()(img)
hyp = model.beam_search(img)
# hmr = vocab.indices2words(hyp.seq)
print(hyp)
