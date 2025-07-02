# WITHOUT PYARROW
# import requests

# url = "http://127.0.0.1:8000/run_eval/"

# files = {
#     'eval_file': open('eval.py', 'rb'),
#     'data_file': open('data.csv', 'rb')
# }
# data = {
#     'model': 'ollama/tinyllama:latest'
# }

# response = requests.post(url, files=files, data=data)

# print(response.status_code)
# print(response.json())


# WITH PYARROW
import requests
import pyarrow.ipc as pa_ipc
import pandas as pd
import io

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)

url = "http://127.0.0.1:8000/run_eval/"

files = {
    'eval_file': open('eval.py', 'rb'),
    'data_file': open('data.csv', 'rb')
}
data = {
    'model': 'ollama/tinyllama:latest'
}

response = requests.post(url, files=files, data=data)

print("Status code:", response.status_code)

if response.status_code == 200:
    # read Arrow stream from binary output
    buffer = io.BytesIO(response.content)
    reader = pa_ipc.open_stream(buffer)
    table = reader.read_all()

    df = table.to_pandas()
    print(df.head())
else:
    try:
        print("Error response:", response.json())
    except Exception:
        print("Raw response:", response.text)
