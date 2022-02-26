import requests 
import os
import json

keyword='d'
i = 0
for file in os.listdir('data'):
	try:
		print(file)
		f = open(f'data/{file}')
		data = json.load(f)
		for r in data['results']:
			try:
				response = requests.get(r['url'])
				file = open(f"images/image_{keyword}{i}.png", "wb")
				file.write(response.content)
				file.close()
				i+=1
			except Exception as ex:
				print(ex)
				continue
	except Exception as ex:
		print(ex)
		continue
