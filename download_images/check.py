
from PIL import Image
import os

img_path = '/home/ubuntu/data2/abstract/'
for f in os.listdir(img_path):
    try:
        x = Image.open(f'{img_path}{f}')
    except Exception as ex:
        print(f'{img_path}{f}', ex)
  

