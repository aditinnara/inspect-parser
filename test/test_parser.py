import requests

url = "http://127.0.0.1:8000/run_eval/"

files = {
    'eval_file': open('eval.py', 'rb'),
    'data_file': open('data.csv', 'rb')
}
data = {
    'model': 'ollama/tinyllama:latest'
}

response = requests.post(url, files=files, data=data)

print(response.status_code)
print(response.json())