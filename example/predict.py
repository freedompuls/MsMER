from bttr.lit_bttr import LitBTTR
from PIL import Image
from torchvision.transforms import ToTensor
from bttr.datamodule import vocab
import os


ckpt = 'E:/Tensorboard_space/Res2_Dense_BTTR_aug_0.5698/version_2/checkpoints/epoch=165-step=421971-val_ExpRate=0.5563.ckpt'
# img_path = 'E:/data_plus/data_BTTR_resezie_16times/2019/UN19wb_1110_em_1021.bmp'
model = LitBTTR.load_from_checkpoint(ckpt)

root = 'D:'
cla = '\\python_workspace\\HMER_text\\predicate_2019\\2019'
cla_path = os.path.join(root, cla)
supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp"]  # 支持的文件后缀类型
images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

image_path = "D:\\python_workspace\\HMER_text\\predicate_2019\\2019\\"

save_txt_path = r'D:\python_workspace\HMER_text\predicate_2019\predict.txt'
predicate_txt_path = r'D:\python_workspace\HMER_text\predicate_2019\2019\caption.txt'

txt_save = open(save_txt_path, 'w')
txt_predicate = open(predicate_txt_path, 'r')

line_predicta = txt_predicate.readlines()  # 这个是 图片名称加上latex 截取 前面名称

# 这个就是可以 按照 caption中的顺序进行 读取预测写入 txt
for i in line_predicta:
    image_name = i.split("\t")[0]  #  获取图片名称
    img_path = image_path + image_name + '.bmp'  # 拼接成图片地址 用于读取图片
    img = Image.open(img_path)
    img = ToTensor()(img)
    hyp = model.beam_search(img)  # 预测结果
    print(hyp)
    image_prdicate = list(image_name)  # 图片名称 list化
    image_prdicate .append('\t')   # 加上 \t 中间四个空格
    image_prdicate .extend(list(hyp))  # 加上list 话的预测结果
    txt_save.write('\n'+''.join(image_prdicate))   # 加入换行符进行拼接 存储
    txt_save.flush()  # 进行实时存储预测结果

txt_save.close()


# 下面这个部分是直接提取文件夹下图片名称路径 来读取预测 写入txt
# image_path = "D:\\python_workspace\\HMER_text\\predicate_2019\\2019"
# lens = len(image_path)+1
# for i in images:
#     img = Image.open(i)
#     img = ToTensor()(img)
#     hyp = model.beam_search(img)
#     print(hyp)
#     image_name = list(i[lens:-4])
#     image_name.append('\t')
#     image_name.extend(list(hyp))
#     txt_save.write(''.join(image_name))
#     txt_save.flush()
#
# txt_save.close()
