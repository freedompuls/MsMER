# 这个是看路径字符串的 切片操作
cla = '\\python_workspace\\HMER_text\\predicate_2019\\2019'
images = '\\python_workspace\\HMER_text\\predicate_2019\\2019\\12385.txt'
a = len(cla)
c = len(images)
# images = list(images)
b = images[a+1:]
b = images[a+1:-4]
print(a,c)
print(b)

# 47 57
# 12385

# 这个部分是读取 caption.txt 查看 图像名称与 latex 中间的间隔内容 现在明白是 四个空格付   \t
predicate_txt_path = r'D:\python_workspace\HMER_text\predicate_2019\2019\caption.txt'
txt_predicate = open(predicate_txt_path, 'r')
line_predicta = txt_predicate.readlines()
print(line_predicta[1])
print(list(line_predicta[1]))
# UN19_1039_em_567	x = \cos L t
# ['U', 'N', '1', '9', '_', '1', '0', '3', '9', '_', 'e', 'm', '_', '5', '6', '7', '\t', 'x', ' ', '=', ' ', '\\', 'c', 'o', 's', ' ', 'L', ' ', 't', '\n']
print(line_predicta[2])
print(list(line_predicta[2]))
print(line_predicta[2].split("\t")[0])  # 使用 \t 来截取 caption中图片名字
# UN19_1038_em_545	- \frac { 3 + z ^ { 2 } } { 8 } - \frac { ( 3 + z ^ { 2 } ) ^ { 2 } } { 3 2 }
# ['U', 'N', '1', '9', '_', '1', '0', '3', '8', '_', 'e', 'm', '_', '5', '4', '5', '\t', '-', ' ', '\\', 'f', 'r', 'a', 'c', ' ', '{', ' ', '3', ' ', '+', ' ', 'z', ' ', '^', ' ', '{', ' ', '2', ' ', '}', ' ', '}', ' ', '{', ' ', '8', ' ', '}', ' ', '-', ' ', '\\', 'f', 'r', 'a', 'c', ' ', '{', ' ', '(', ' ', '3', ' ', '+', ' ', 'z', ' ', '^', ' ', '{', ' ', '2', ' ', '}', ' ', ')', ' ', '^', ' ', '{', ' ', '2', ' ', '}', ' ', '}', ' ', '{', ' ', '3', ' ', '2', ' ', '}', '\n']