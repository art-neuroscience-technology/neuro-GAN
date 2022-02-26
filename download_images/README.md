Download data from Openverse 
---------------------------- 

# Register key 
```bash
curl \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"name": "Neurofeedback", "description": "Access Openverse API", "email": "{mail}"}' \
  "https://api.openverse.engineering/v1/auth_tokens/register"
```

**Example response:**

```json
  {"client_id":"{client_id}","client_secret":"{secret}","name":"Neurons","msg":"Check your email for a verification link."}
```

# Get token

```bash
curl \
  -X POST \
  -d "client_id={client_id}&client_secret={secret}&grant_type=client_credentials" \
  "https://api.openverse.engineering/v1/auth_tokens/token/"    
```

**Example response:**
```json
  {"access_token": "1vb8DgNOWRbkn6n6nDQNk1kBFSmXYQ", "expires_in": 36000, "token_type": "Bearer", "scope": "read write"} 
```

# Download page 
```bash
curl -H "Authorization: Bearer 1vb8DgNOWRbkn6n6nDQNk1kBFSmXYQ" "https://api.openverse.engineering/v1/images/?q=neuron?page=2" > neurons2.json 
```

# FILES 
1. This scripts have been used to download data from [Openverse](https://wordpress.org/openverse/) 

- [dowbload_openverse.py](dowbload_openverse.py)
- [dowbload_pages.py](dowbload_pages.py)

2. This scripts have been used to download data from drive
- [BingImages.ipynb](https://colab.research.google.com/drive/1kBwZwtxD4tYsnuGLcc3_GHqqspuBc8QV?usp=sharing) Google Colab for download images from [Bing](https://www.bing.com/) browser 
- [dowbload_drive.py](dowbload_drive.py) Download zip from drive (result from downloading all images)

3. Validate images 
- [check.py](check.py) Script to check that images are valid 
