# WITHOUT PYARROW
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.responses import JSONResponse
# import tempfile, os, subprocess
# from inspect_ai.log import read_eval_log_sample_summaries, read_eval_log, read_eval_log_samples

# # figure out env vars
# import os
# from dotenv import load_dotenv
# load_dotenv()

# # Create API instance
# app = FastAPI()

# # run-eval endpoint lets the user provide eval.py, data.csv, and the model name
# # These are required inputs
# @app.post("/run_eval/")
# async def evaluate(
#     eval_file: UploadFile = File(...),
#     data_file: UploadFile = File(...),
#     model: str = Form(...)
# ):
#     try:
#         # Create temp dir 
#         with tempfile.TemporaryDirectory() as tmpdir:
#             # Save eval.py and data.csv as temp files in that temp dir
#             eval_path = os.path.join(tmpdir, "eval.py")
#             data_path = os.path.join(tmpdir, "data.csv")

#             with open(eval_path, "wb") as f:
#                 f.write(await eval_file.read())

#             with open(data_path, "wb") as f:
#                 f.write(await data_file.read())

#             # Run inspect eval on the provided evaluation
#             result = subprocess.run(
#                 ["inspect", "eval", "eval.py", "--model", model],
#                 cwd=tmpdir,
#                 capture_output=True,
#                 text=True
#             )

#             # Handle errors from running `inspect eval`
#             if result.returncode != 0:
#                 raise HTTPException(status_code=500, detail={
#                     "message": "inspect eval failed",
#                     "stderr": result.stderr,
#                     "stdout": result.stdout
#                 })

#             # Navigate to logs directory
#             logs_dir = os.path.join(tmpdir, "logs")

#             # Get the latest log file based on modification time
#             log_files = sorted(
#                 [os.path.join(logs_dir, f) for f in os.listdir(logs_dir)],
#                 key=os.path.getmtime,
#                 reverse=True
#             )

#             # Make sure there's actually a log. If so, grab the first (latest) log.
#             if not log_files:
#                 raise HTTPException(status_code=500, detail="No log files found.")
#             log_file = log_files[0]

#             # Open the log if there wasn't an error
#             log = read_eval_log(log_file)
#             if log.status != "success":
#                 raise HTTPException(status_code=500, detail="Evaluation failed: " + log.status)


#             # TODO: make this prettier, idk how we want the output to look
            
#             # Do this if we want to return only sample summaries...
#             # Return the sample summaries as a JSON dump
#             summaries = read_eval_log_sample_summaries(log_file)
#             return JSONResponse(content=[s.model_dump() for s in summaries])

#             # # Do this if we want to return the entire sample list...
#             # # Use read_eval_log_samples to get all samples (generator)
#             # samples_generator = read_eval_log_samples(log_file)
#             # # Convert generator to list of dicts
#             # samples_list = [sample.model_dump() for sample in samples_generator]
#             # return JSONResponse(content=samples_list)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# WITH PYARROW
from fastapi.responses import StreamingResponse
import pyarrow as pa
import pandas as pd
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import tempfile, os, subprocess
from inspect_ai.log import read_eval_log_sample_summaries, read_eval_log, read_eval_log_samples

# figure out env vars
import os
from dotenv import load_dotenv
load_dotenv()

# Create API instance
app = FastAPI()

# run-eval endpoint lets the user provide eval.py, data.csv, and the model name
# These are required inputs
@app.post("/run_eval/")
async def evaluate(
    eval_file: UploadFile = File(...),
    data_file: UploadFile = File(...),
    model: str = Form(...)
):
    try:
        # Create temp dir 
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save eval.py and data.csv as temp files in that temp dir
            eval_path = os.path.join(tmpdir, "eval.py")
            data_path = os.path.join(tmpdir, "data.csv")
            
            with open(eval_path, "wb") as f:
                f.write(await eval_file.read())

            with open(data_path, "wb") as f:
                f.write(await data_file.read())

            # Run inspect eval on the provided evaluation and model
            result = subprocess.run(
                ["inspect", "eval", "eval.py", "--model", model],
                cwd=tmpdir,
                capture_output=True,
                text=True
            )

            # Handle failures from inspect eval
            if result.returncode != 0:
                raise HTTPException(status_code=500, detail={
                    "message": "inspect eval failed",
                    "stderr": result.stderr,
                    "stdout": result.stdout
                })

            # Navigate to logs directory and grab most recent log
            logs_dir = os.path.join(tmpdir, "logs")
            log_files = sorted(
                [os.path.join(logs_dir, f) for f in os.listdir(logs_dir)],
                key=os.path.getmtime,
                reverse=True
            )

            # Open the log if there wasn't an error
            if not log_files:
                raise HTTPException(status_code=500, detail="No log files found.")
            log_file = log_files[0]

            # Make sure evaluation succeeded before getting the samples
            log = read_eval_log(log_file)
            if log.status != "success":
                raise HTTPException(status_code=500, detail="Evaluation failed: " + log.status)

            # Get summaries -- there will be 1 per sample
            summaries = read_eval_log_sample_summaries(log_file)

            # Build a dict of outputs by sample ID
            outputs_by_id = {}
            for sample in read_eval_log_samples(log_file):
                output_texts = [c.message.content for c in sample.output.choices] if sample.output else None
                outputs_by_id[str(sample.id)] = output_texts  

            # Combine each summary with its output_texts
            dicts = []
            for summary in summaries:
                d = summary.model_dump()
                sample_id = str(summary.id)
                d["output_texts"] = outputs_by_id.get(sample_id)
                dicts.append(d)

            df = pd.json_normalize(dicts)  
            # Convert to Arrow Table
            table = pa.Table.from_pandas(df)

            # Serialize to Arrow IPC stream
            sink = io.BytesIO()
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
            sink.seek(0)
            return StreamingResponse(
                content=sink,
                media_type="application/vnd.apache.arrow.stream",
                headers={
                    "Content-Disposition": "attachment; filename=summaries.arrow"
                }
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
