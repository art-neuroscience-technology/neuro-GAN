import gdown

url = 'https://drive.google.com/uc?id={id}'
output = 'abstract2.zip'
gdown.download(url, output, quiet=False)