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

            # Run inspect eval on the provided evaluation
            result = subprocess.run(
                ["inspect", "eval", "eval.py", "--model", model],
                cwd=tmpdir,
                capture_output=True,
                text=True
            )

            # Handle errors from running `inspect eval`
            if result.returncode != 0:
                raise HTTPException(status_code=500, detail={
                    "message": "inspect eval failed",
                    "stderr": result.stderr,
                    "stdout": result.stdout
                })

            # Navigate to logs directory
            logs_dir = os.path.join(tmpdir, "logs")

            # Get the latest log file based on modification time
            log_files = sorted(
                [os.path.join(logs_dir, f) for f in os.listdir(logs_dir)],
                key=os.path.getmtime,
                reverse=True
            )

            # Make sure there's actually a log. If so, grab the first (latest) log.
            if not log_files:
                raise HTTPException(status_code=500, detail="No log files found.")
            log_file = log_files[0]

            # Open the log if there wasn't an error
            log = read_eval_log(log_file)
            if log.status != "success":
                raise HTTPException(status_code=500, detail="Evaluation failed: " + log.status)

            # Do this if we want to return only sample summaries...
            # Return the sample summaries as a JSON dump
            summaries = read_eval_log_sample_summaries(log_file)
            return JSONResponse(content=[s.model_dump() for s in summaries])

            # # Do this if we want to return the entire sample list...
            # # Use read_eval_log_samples to get all samples (generator)
            # samples_generator = read_eval_log_samples(log_file)
            # # Convert generator to list of dicts
            # samples_list = [sample.model_dump() for sample in samples_generator]
            # return JSONResponse(content=samples_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





