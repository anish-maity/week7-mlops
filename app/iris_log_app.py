from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import time
import json
import logging

# --- OpenTelemetry Imports ---
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# --- Configure OpenTelemetry with explicit service + project ---
PROJECT_ID = "fresh-bloom-472914-s8"

resource = Resource.create({
    "service.name": "iris-log-service",
    "gcp.project_id": PROJECT_ID,
    "cloud.provider": "gcp",
    "cloud.platform": "gcp_kubernetes_engine",
    "k8s.cluster.name": "demo-log-ml-cluster",
})

provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)

# Use ADC (Workload Identity handles auth)
cloud_trace_exporter = CloudTraceSpanExporter(project_id=PROJECT_ID)
provider.add_span_processor(BatchSpanProcessor(cloud_trace_exporter))

# Create tracer instance
tracer = trace.get_tracer(__name__)

# --- Structured Logging Setup ---
logger = logging.getLogger("iris-ml-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()

formatter = logging.Formatter(json.dumps({
    "severity": "%(levelname)s",
    "message": "%(message)s",
    "timestamp": "%(asctime)s"
}))
handler.setFormatter(formatter)
logger.addHandler(handler)

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Input Schema ---
class Input(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# --- App State ---
app_state = {"is_ready": False, "is_alive": True}


@app.on_event("startup")
async def startup_event():
    global model
    time.sleep(2)
    model = joblib.load("model.joblib")
    app_state["is_ready"] = True
    logger.info("Model loaded successfully")


# --- Health & Readiness Probes ---
@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=500)


@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=503)


# --- Request Timing Middleware ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response


# --- Global Exception Handler ---
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error", "trace_id": trace_id})


# --- Prediction Endpoint ---
@app.post("/predict")
async def predict(input: Input, request: Request):
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")

        try:
            import pandas as pd
            # Use the same feature names as the model was trained with
            input_data = pd.DataFrame([[
                input.sepal_length,
                input.sepal_width,
                input.petal_length,
                input.petal_width
            ]], columns=[
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width"
            ])

            result = model.predict(input_data)[0]  # e.g. "setosa"
            latency = round((time.time() - start_time) * 1000, 2)

            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "input": input_data.to_dict(orient="records"),
                "result": result,
                "latency_ms": latency,
                "status": "success"
            }))

            return {
                "prediction": result,
                "trace_id": trace_id,
                "latency_ms": latency
            }

        except Exception as e:
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail="Prediction failed")
