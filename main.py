from collections import defaultdict
from fastapi.responses import StreamingResponse
import pyarrow as pa
import pandas as pd
import numpy as np
import io, datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import tempfile, os, subprocess, logging
from inspect_ai.log import read_eval_log, read_eval_log_samples
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Mock Vault store schema
VAULT = {
    "models": {
        "ollama/llama3:latest": {
            "model_backend": "ollama",
            "model_key": "ollama/llama3:latest",
            "env_vars": {
                "OLLAMA_BASE_URL": "http://ollama-dev:11434/v1"
            }
        },
        "mistral": {
            "model_backend": "vllm",
            "model_key": "mistral",
            "env_vars": {
                "VLLM_API_BASE": "http://localhost:8000"
            }
        }
    }, 
    
    "api_keys": {
        "openai": "sk-xxx",
        "gemini": "gemini-key",
        "anthropic": "anthropic-key"
    }
}


# Create FastAPI instance
app = FastAPI()

@app.post("/run_eval/")
async def evaluate(
    eval_file: UploadFile = File(...),
    data_file: UploadFile = File(...),
    model: str = Form(...)
):
    logger.info(f"Received request to /run_eval/ with model: {model}")

    # If we test with echo command
    if model == "echo":
        # just return input CSV + score column with 100.0 for all rows
        try:
            logger.info("Echo model detected, skipping inspect, returning input data with success score")

            data_bytes = await data_file.read()
            df = pd.read_csv(io.BytesIO(data_bytes))
            logger.info(f"Input CSV loaded, shape: {df.shape}")

            df["score"] = 100.0
            df["score"] = df["score"].astype("float32")
            logger.info("Added 'score' column with 100.0 float32 for all rows")

            table = pa.Table.from_pandas(df)


            sink = io.BytesIO()
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
            sink.seek(0)

            logger.info("Returning Arrow stream response for echo model with scores")
            return StreamingResponse(
                content=sink,
                media_type="application/vnd.apache.arrow.stream",
                headers={"Content-Disposition": "attachment; filename=echo_with_scores.arrow"}
            )
        except Exception as e:
            logger.exception("Failed to process echo model")
            raise HTTPException(status_code=500, detail=f"Echo model processing error: {e}")

    
    # Otherwise prepare an inspect call
    try:
        # Look up model config from vault
        model_config = VAULT["models"].get(model)
        if not model_config:
            logger.error(f"Unknown model requested: {model}")
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
        logger.info(f"Model config retrieved: {model_config}")

        # Prepare environment variables for subprocess
        env = os.environ.copy()
        env["MODEL_BACKEND"] = model_config["model_backend"]
        env["MODEL_KEY"] = model_config["model_key"]
        env.update(model_config.get("env_vars", {}))

        env["OPENAI_API_KEY"] = VAULT["api_keys"]["openai"]
        env["GEMINI_API_KEY"] = VAULT["api_keys"]["gemini"]
        env["ANTHROPIC_API_KEY"] = VAULT["api_keys"]["anthropic"]
        logger.info("Environment variables set for subprocess.")

        # Create temp dir 
            # with tempfile.TemporaryDirectory() as tmpdir:
        
        # Using permanent dir for now/testing
        logs_root = "./eval_logs"
        os.makedirs(logs_root, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tmpdir = os.path.join(logs_root, f"run_{timestamp}")
        os.makedirs(tmpdir, exist_ok=True)
        logger.info(f"Created persistent directory: {tmpdir}")

        # Save eval.py and data.csv
        eval_path = os.path.join(tmpdir, "eval.py")
        data_path = os.path.join(tmpdir, "data.csv")
        logger.info(f"Saving eval.py to {eval_path}")
        with open(eval_path, "wb") as f:
            f.write(await eval_file.read())
        logger.info(f"Saving data.csv to {data_path}")
        with open(data_path, "wb") as f:
            f.write(await data_file.read())

        # Run inspect eval command
        cmd = ["inspect", "eval", "eval.py", "--model", model]
        logger.info(f"Running subprocess: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True,
            env=env
        )
        logger.info(f"Subprocess finished with return code {result.returncode}")
        if result.stdout:
            logger.info(f"Subprocess stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Subprocess stderr:\n{result.stderr}")

        if result.returncode != 0:
            logger.error("inspect eval failed")
            raise HTTPException(status_code=500, detail={
                "message": "inspect eval failed",
                "stderr": result.stderr,
                "stdout": result.stdout
            })

        # Find log files
        logs_dir = os.path.join(tmpdir, "logs")
        log_files = sorted(
            [os.path.join(logs_dir, f) for f in os.listdir(logs_dir)],
            key=os.path.getmtime,
            reverse=True
        )
        logger.info(f"Found {len(log_files)} log files")

        if not log_files:
            logger.error("No log files found after evaluation.")
            raise HTTPException(status_code=500, detail="No log files found.")

        # Get the log file  
        log_file = log_files[0]
        logger.info(f"Using log file: {log_file}")

        # Return relevant info from log
        dicts, metrics = return_summary_dicts(log_file)

        # Nomalize dict and convert to arrow table
        df = pd.json_normalize(dicts)
        logger.info(f"Normalized dataframe shape: {df.shape}")

        table = pa.Table.from_pandas(df)
        logger.info("Converted dataframe to Arrow table")


        # Add stats as metadata
        table = table.replace_schema_metadata({
            **(table.schema.metadata or {}),
            b"metrics": str(metrics).encode("utf-8")
        })

        sink = io.BytesIO()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        sink.seek(0)

        # Return arrow table as a stream
        logger.info("Returning Arrow stream response")
        return StreamingResponse(
            content=sink,
            media_type="application/vnd.apache.arrow.stream",
            headers={
                "Content-Disposition": "attachment; filename=summaries.arrow"
            }
        )

    except Exception as e:
        logger.exception("Unhandled exception in /run_eval/")
        raise HTTPException(status_code=500, detail=str(e))


# Given a log file, extract the relevant info
def return_summary_dicts(log_file):
    log = read_eval_log(log_file)
    if log.status != "success":
        logger.error(f"Evaluation failed with status: {log.status}")
        raise HTTPException(status_code=500, detail="Evaluation failed: " + log.status)



    # This will be a list of the results of each scorer. We're currently only supporting one scorer per eval.
    for score in log.results.scores:
        metrics = {}
        for metric_type, metric_info in score.metrics.items():
            metrics[metric_type] = metric_info.value

    dicts = []

    for sample in read_eval_log_samples(log_file):
        # Normalize input
        if isinstance(sample.input, list):
            input_text = " ".join(m.content for m in sample.input if hasattr(m, "content"))
        else:
            input_text = sample.input

        # Extract output 
        if sample.output and sample.output.choices:
            output_texts = [choice.message.content for choice in sample.output.choices if hasattr(choice.message, "content")]
        else:
            output_texts = None

        # Extract choices (list of strings) if any
        choices = sample.choices if sample.choices is not None else None

        # Extract target (string or list)
        target = sample.target if sample.target is not None else None

        # Extract messages (list of ChatMessage) => convert to list of strings for readability
        if sample.messages:
            messages_text = []
            for msg in sample.messages:
                if hasattr(msg, "content"):
                    messages_text.append(msg.content)
                else:
                    messages_text.append(str(msg))
        else:
            messages_text = None

        # Extract scores dictionary (serialize Score objects to dict if needed)
        if sample.scores:
            scores = {k: v.model_dump() if isinstance(v, BaseModel) else v for k, v in sample.scores.items()}
        else:
            scores = None
        # print("scores: ", scores)
        for scorer, score_info in scores.items():
            score = np.float32(1.0 if score_info['value'] == "C" else 0.0)

        # Metadata (dict)
        metadata = sample.metadata if sample.metadata else None

        # Model name from output.model (str)
        model_name = sample.output.model if sample.output and hasattr(sample.output, "model") else None

        dicts.append({
            "input": input_text,
            "output": output_texts,
            "choices": choices,
            "target": target,
            "messages": messages_text,
            "metadata": metadata,
            "score": score,
            "model": model_name
        })

    return dicts, metrics
