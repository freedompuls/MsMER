from bttr.lit_bttr import LitBTTR
from PIL import Image
from torchvision.transforms import ToTensor
import os


ckpt = 'E:/Tensorboard_space/Res2_Dense_BTTR_aug_0.5698/version_2/checkpoints/epoch=165-step=421971-val_ExpRate=0.5563.ckpt'
img_path = 'D:\\python_workspace\\HMER_text\\predicate_2019\\2019\\ISICal19_1201_em_750.bmp'


strip = ' '
save_txt_path = r'D:\python_workspace\HMER_text\predicate_2019\predict.txt'
txt_save = open(save_txt_path, 'w')

image_path = "D:\\python_workspace\\HMER_text\\predicate_2019\\2019"
lens = len(image_path)+1

model = LitBTTR.load_from_checkpoint(ckpt)
img = Image.open(img_path)
img = ToTensor()(img)
hyp = model.beam_search(img)
print(hyp)
image_name = list(img_path[lens:-4])
print(image_name)
image_name_len = len(image_name)
image_name.append('\t')
print(image_name)
# hyp = list(hyp)
image_name.extend(list(hyp))
# image_name = image_name + hyp
#  + 法或者 extend() 都可以
print(image_name)
# txt_save.write(''.join(image_name))
# txt_save.close()

# - \frac { 1 } { 1 9 2 }
# ['I', 'S', 'I', 'C', 'a', 'l', '1', '9', '_', '1', '2', '0', '1', '_', 'e', 'm', '_', '7', '5', '0']
# ['I', 'S', 'I', 'C', 'a', 'l', '1', '9', '_', '1', '2', '0', '1', '_', 'e', 'm', '_', '7', '5', '0', '\t']
# ['I', 'S', 'I', 'C', 'a', 'l', '1', '9', '_', '1', '2', '0', '1', '_', 'e', 'm', '_', '7', '5', '0', '\t', '-', ' ', '\\', 'f', 'r', 'a', 'c', ' ', '{', ' ', '1', ' ', '}', ' ', '{', ' ', '1', ' ', '9', ' ', '2', ' ', '}']
