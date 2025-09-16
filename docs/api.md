# API Documentation

## Video Processing API

### POST /api/v1/process/auto

**Summary:** Automatically enhance video quality using SOTA models.

**Description:** This endpoint analyzes the input video and automatically selects the best enhancement strategy based on detected degradations and content characteristics.

**Request Body:**

-   `file`: The video file to process.
-   `request`: A JSON object with the following optional fields:
    -   `vsr_strategy`: `auto`, `vsrm`, `seedvr2`, `ditvr`, `fast_mamba_vsr`
    -   `latency_class`: `strict`, `standard`, `flexible`
    -   `quality_tier`: `fast`, `balanced`, `high`, `ultra`
    -   `target_fps`: (integer) 15-120
    -   `target_resolution`: (string) e.g., '1920x1080', '4K'
    -   `scale_factor`: (float) 1.0-4.0
    -   `allow_diffusion`: (boolean)
    -   `allow_zero_shot`: (boolean)
    -   `enable_face_expert`: (boolean)
    -   `enable_hfr`: (boolean)
    -   `enable_temporal_consistency`: (boolean)

**Responses:**

-   `200 OK`: Returns a `ProcessingResponse` object with the job ID and status.
-   `422 Unprocessable Entity`: Invalid input parameters.
-   `500 Internal Server Error`: An unexpected error occurred.

### GET /api/v1/process/job/{job_id}

**Summary:** Get detailed status information for a processing job.

**Responses:**

-   `200 OK`: Returns a `JobStatusResponse` object with the job status and progress.
-   `404 Not Found`: The job ID was not found.
