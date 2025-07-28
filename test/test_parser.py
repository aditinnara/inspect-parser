import requests
import pyarrow.ipc as pa_ipc
import pandas as pd
import io

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)

url = "http://127.0.0.1:8000/run_eval/"

eval_to_test = "bisexual"

files = {
    'eval_file': open(eval_to_test + '/task.py', 'rb'),
    'data_file': open(eval_to_test + '/data.csv', 'rb')
}
data = {
    'model': 'ollama/llama3:latest'
    # 'model': 'echo'
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

    metadata = table.schema.metadata
    accuracy = metadata.get(b"accuracy", b"").decode("utf-8")
    stderr = metadata.get(b"stderr", b"").decode("utf-8")

    print(f"\nAccuracy (from metadata): {accuracy}")
    print(f"StdErr (from metadata): {stderr}")

else:
    try:
        print("Error response:", response.json())
    except Exception:
        print("Raw response:", response.text)
