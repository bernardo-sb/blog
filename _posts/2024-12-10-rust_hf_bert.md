---
title:  "Serving Deep Learning Models with Rust (and comparing it to Python)"
mathjax: true
layout: post
categories: media
---

Building a service to serve machine learning models, such as a BERT-based embedding generator, requires careful consideration of factors like performance, ease of development, and maintainability. This article explores two implementations of such a service—one in **Rust** and the other in **Python**—and highlights their design choices, strengths, and trade-offs.

To explore the efficiency of serving machine learning models, I conducted a benchmark comparison between **Python** and **Rust** implementations. My initial hypothesis was that **Rust**, with its reputation for high performance and low-level control, would significantly outperform **Python**.

Additionally, I aimed to investigate how different concurrency mechanisms—such as `RwLock` and `Mutex` — and the choice between sharing or not sharing the model state among workers would influence performance.

---

## Rust Implementation

To explore the efficiency of serving machine learning models, I conducted a benchmark comparison between Python and Rust implementations. My initial hypothesis was that Rust, with its reputation for high performance and low-level control, would significantly outperform Python. Additionally, I aimed to investigate how different concurrency mechanisms—such as `RwLock` and `Mutex` — and the choice between sharing or not sharing the model state among workers would influence performance.

The results revealed nuanced insights. While `RwLock` (`tokio`) demonstrated better performance than `Mutex` in Rust as expected, and attributable to its asynchronous nature and the implementation which required safe concurrent reads (`Mutex` doesn't distinguish between reads and writes).

### Overview

The Rust implementation uses the **Actix Web** framework for handling HTTP requests and the **Candle** library to serve a BERT model. The primary components include:

1. **Model Initialization**:
   - The `model::Bert` struct encapsulates the BERT model and tokenizer.
   - It leverages the `candle_core` and `hf_hub` libraries for tensor computations and downloading pre-trained models.

2. **Concurrency**:
   - Uses `tokio::sync::RwLock` for safe concurrent access to the model during prediction.
   - The Actix Web server is configured with 10 workers to handle multiple requests concurrently.

3. **Request Handling**:
   - The `/predict` endpoint processes input text, generates embeddings, and responds in JSON format.
   - Middleware logs the payload size and request processing time.

4. **Performance Optimizations**:
   - The model is loaded once during server startup and cached in memory.
   - Model tokenizer doesn't apply padding or truncation.
   - Uses memory-mapped tensors for efficient loading.
   - Immutable model allows the use `RwLock`'s `read` method.

### Code

#### **main.rs - Setting Up the Web Server**

##### **Modules and Imports**
```rust
mod model;
use actix_web::{self, Responder};
use actix_web::{
    body::MessageBody,
    dev::{ServiceRequest, ServiceResponse},
    middleware::{from_fn, Next},
    Error,
};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tokio::sync::RwLock;
```

Here, the code imports modules and libraries necessary for creating the web API. Key highlights:
- **`actix_web`** powers the web server and request routing.
- **`tokio::sync::RwLock`** is used to enable thread-safe, asynchronous shared access to the BERT model.
- **`serde`** facilitates serialization and deserialization of request/response payloads.
- **`model`** is the custom module containing the BERT implementation.

#### **Data Structures**
```rust
#[derive(Deserialize)]
struct Prompt {
    text: String,
}

#[derive(Serialize)]
struct Embeddings {
    ys: Vec<Vec<Vec<f32>>>,
}

struct AppModel {
    bert: RwLock<model::Bert>,
}
```

- **`Prompt`**: Deserializes the user-provided input (`text`).
- **`Embeddings`**: Serializes the response (a multi-dimensional vector of embeddings).
- **`AppModel`**: A container for the BERT model wrapped in an asynchronous `RwLock`, allowing safe concurrent reads.

---

#### **Prediction Endpoint**
```rust
#[actix_web::post("/predict")]
async fn predict(
    prompt: actix_web::web::Json<Prompt>,
    model: actix_web::web::Data<AppModel>,
) -> impl Responder {
    let model = model.bert.read().await;

    println!("Predicting: {:?}", prompt.text);

    let ys = model
        .predict(prompt.text.clone())
        .unwrap()
        .to_vec3()
        .unwrap();

    actix_web::web::Json(Embeddings { ys })
}
```

- **Endpoint Logic**:
  - Reads the shared `BERT` model.
  - Extracts and logs the input text from the `Prompt`.
  - Passes the input text to the `predict` method of the BERT model.
  - Converts the model output to a JSON-compatible structure for the response.

- **Concurrency**: The use of `RwLock` ensures multiple requests can read the model concurrently without conflicts.

---

#### **Middleware for Logging**
```rust
async fn middleware_time_elapsed(
    req: ServiceRequest,
    next: Next<impl MessageBody>,
) -> Result<ServiceResponse<impl MessageBody>, Error> {
    let payload_size = req
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);

    println!("Payload size: {:?}", payload_size);

    let now = std::time::Instant::now();

    let res = next.call(req).await;

    println!("Request took: {:?}", now.elapsed());

    res
}
```

- Logs the size of the payload and the time taken to process each request.
- Acts as a general-purpose performance tracking mechanism.

---

#### **Server Initialization**
```rust
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let model = actix_web::web::Data::new(AppModel {
        bert: RwLock::new(model::Bert::new(None, None, false, false).unwrap()),
    });

    println!("Current PID: {:?}", std::process::id());

    actix_web::HttpServer::new(move || {
        actix_web::App::new()
            .app_data(model.clone())
            .wrap(from_fn(middleware_time_elapsed))
            .service(predict)
    })
    .workers(10)
    .bind(("127.0.0.1", 8000))?
    .run()
    .await
}
```

- **Model Initialization**:
  - The `AppModel` is instantiated with the BERT model, which is created using the `model::Bert::new` function.
- **Middleware and Routes**:
  - Attaches the logging middleware and prediction endpoint to the application.
- **Concurrency**:
  - Spawns 10 workers for handling incoming requests.

---

### **model.rs - Implementing the BERT Model**

#### **Imports**

```rust
use candle_core::{Device, Tensor};
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use anyhow::{Error as E, Result};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde_json;
use tokenizers::{
    DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper,
    Tokenizer, TokenizerImpl,
};
```

- **`candle_transformers`**: A Rust library for transformers, including BERT.
- **`hf_hub`**: Interacts with Hugging Face Hub to download model artifacts.
- **`tokenizers`**: Provides efficient tokenization.

---

### **BERT Model Definition**

```rust
pub struct Bert {
    model: BertModel,
    tokenizer: TokenizerImpl<
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >,
    device: Device,
}
```

The `Bert` struct encapsulates:
- The transformer model (`BertModel`).
- The tokenizer for input text.
- The target computation device (CPU or GPU).

---

### **Loading the Model**
```rust
impl Bert {
    pub fn new(
        model_id: Option<String>,
        revision: Option<String>,
        use_torch: bool,
        approximate_gelu: bool,
    ) -> Result<Bert> {
        let device = Device::Cpu;

        // Set the model and revision
        let default_model = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let default_revision = "refs/pr/21".to_string();
        let (model_id, revision) = match (model_id.to_owned(), revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);

        // Get files for loading the model
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = if use_torch {
                api.get("pytorch_model.bin")?
            } else {
                api.get("model.safetensors")?
            };
            (config, tokenizer, weights)
        };

        let config = std::fs::read_to_string(config_filename)?;
        let mut config: Config = serde_json::from_str(&config)?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        // Set padding and truncation to None
        let tokenizer = tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(E::msg)
            .unwrap();

        let tokenizer = &*tokenizer; // Get as immutable reference

        let vb = if use_torch {
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
        };
        if approximate_gelu {
            config.hidden_act = HiddenAct::GeluApproximate;
        }

        let model = BertModel::load(vb, &config)?;
        let tokenizer = tokenizer.clone();

        Ok(Bert {
            model: model,
            tokenizer: tokenizer,
            device: device,
        })
    }
...
```

- Downloads configuration, tokenizer, and weights files from Hugging Face.
- Loads the model and tokenizer into memory.
-  Get immutable tokeninzer:`let tokenizer = &*tokenizer;`. This allows the model to be accessed concurrently, if it were mutable that wouldn't be possible,

---

### **Making Predictions**

```rust
// impl Bert {
...
    pub fn predict(&self, prompt: String) -> Result<Tensor> {
        let start = std::time::Instant::now();

        // Tokenization
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;

        println!("Loaded and encoded {:?}", start.elapsed());

        // Forward pass
        let ys = self.model.forward(&token_ids, &token_type_ids, None)?;
        println!("Forward pass took {:?}", start.elapsed());

        Ok(ys)
    }
```

- Tokenizes the input text into token IDs.
- Performs a forward pass through the model to generate embeddings.


### Key Rust Features

The case for Rust lies on top of the following features, being the last one crucial in backend development:

- **Concurrency**: Rust's ownership model ensures thread safety with minimal runtime overhead.
- **Performance**: Close-to-hardware control offers high efficiency, especially for CPU-bound tasks.
- **Type Safety**: The compiler enforces strict type checks, reducing runtime errors.

---

## Python Implementation

### Overview

The Python implementation uses **FastAPI** as the web framework and **Sentence Transformers** for serving the BERT model. Key components include:

1. **Model Initialization**:
   - The model is loaded globally during startup using the `SentenceTransformer` library.
   - It uses the same `all-MiniLM-L6-v2` model and revision as the Rust implementation.

2. **Request Handling**:
   - The `/predict` endpoint processes the input, generates embeddings, and responds in JSON format.
   - Includes error handling for scenarios where the model is not loaded or the input is invalid.

3. **Ease of Use**:
   - Python's high-level abstractions make the implementation concise and beginner-friendly.

### Key Python Features

Python's biggest pros are its simplicity and large community:

- **Rapid Development**: FastAPI simplifies API development with its declarative syntax and automatic OpenAPI documentation generation.
- **Flexibility**: Dynamic typing and rich ML ecosystems (e.g., Sentence Transformers) speed up prototyping and integration.
- **Readable Code**: The implementation is straightforward and easy to maintain.

### Code


#### **The Anatomy of the Service**

##### **Imports and Setup**
```python
import datetime
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
```

- **`FastAPI`**: A Python web framework designed for fast, asynchronous APIs.
- **`APIRouter`**: Allows modular routing, making it easy to build and extend APIs.
- **`SentenceTransformer`**: Provides sentence embeddings using pre-trained models.
- **`datetime`**: Tracks request processing time for debugging and performance monitoring.

---

#### **Loading the Model**

```python
model = None

def load_model():
    global model
    model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2',
        revision="refs/pr/21",
        device="cpu"
    )
```

- **Global Model Initialization**: The `model` variable is set as `None` initially. The **`load_model`** function loads the model from Hugging Face's repository.
- **Model Revision**: Pinning to a specific `revision` ensures consistency in behavior across deployments and allows a directo comparison to Rust's implementation.
- **Device**: For simplicity, this code uses CPU. For better performance in production, consider leveraging a GPU.

---

#### **API Schema Definition**
FastAPI uses **Pydantic** models for request/response validation.

```python
class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
```

- **`EmbeddingRequest`**: Expects a single string (`text`) as input.
- **`EmbeddingResponse`**: Returns a nested list of floats, representing the embeddings.

---

#### **Embedding Endpoint**

```python
@router.post("/predict", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    start = datetime.datetime.now()

    global model

    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        print(f"Generating embeddings for: {request.text}...")

        embeddings = model.encode([request.text], convert_to_tensor=False)

        end = datetime.datetime.now()
        print(f"Time elapsed: {(end - start).total_seconds() * 1000} ms")

        return EmbeddingResponse(embeddings=embeddings)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating embeddings: {e}")
```

##### Key Features:
1. **Timing for Performance Monitoring**:
   - Logs the time taken to generate embeddings in milliseconds.
   - Helpful for debugging latency issues.

2. **Global Model Access**:
   - Ensures the pre-loaded model is used for predictions.
   - Returns an HTTP 500 error if the model hasn't been loaded (safety check).

3. **Embedding Generation**:
   - **`model.encode`** converts the input text into embeddings.
   - Outputs a list of float lists, one for each input sentence (batch size = 1 here).

4. **Error Handling**:
   - Catches and reports errors gracefully with HTTP 500 responses.

---

#### **Application Setup**

The `run()` function ties everything together.

```python
def run() -> FastAPI:
    load_model()

    app = FastAPI()
    app.include_router(router)

    return app
```

- **Model Loading**: The `load_model` function is called once during app initialization.
- **Modular Routing**: The router (`router`) is attached to the app, ensuring clean separation of concerns.

---

#### **Running**
The last step is running the application using Gunicorn:

```bash
gunicorn --preload -w 10 -k uvicorn.workers.UvicornWorker server:app --bind 0.0.0.0:8000
```

- **Preload (`--preload`)**:  Loads the application code before forking workers, reducing memory usage through shared memory and enabling faster startup times for worker processes.
- **Workers (`-w 10`)**: Spawns 10 worker processes to handle requests concurrently.
- **Uvicorn Worker (`-k uvicorn.workers.UvicornWorker`)**: Leverages Uvicorn’s high-performance asynchronous worker.
- **Binding**: Exposes the app on `0.0.0.0:8000` for external access.

---

## Comparison

Both implementation were tested by making multiple request with the same payload.

| Aspect                  | Rust Implementation                     | Python Implementation              |
|-------------------------|-----------------------------------------|------------------------------------|
| **Performance**         | Highly optimized, suitable for low-latency requirements. | Slower due to Python's GIL and overhead. |
| **Concurrency**         | Uses `tokio::RwLock` for safe, concurrent model access. | Relies on FastAPI's asynchronous nature but is limited by Python's threading model. |
| **Model Loading**       | Memory-mapped tensors and selective format loading (Torch or safetensors). | Straightforward loading with the `SentenceTransformer` library. |
| **Development Speed**   | Requires more effort due to low-level control. | Faster due to Python’s simplicity and mature libraries. |
| **Error Handling**      | Compile-time checks prevent many bugs. | Runtime error handling is flexible but prone to oversight. |
| **Deployment**          | Compiled binary with no dependencies.  | Requires Python runtime and dependencies. |
| **Maintainability**     | Verbose codebase with strong type constraints. | Easier to modify and extend for newcomers. |

---

### Analyzing the Benchmarks: Rust vs. Python

#### **Rust Implementation**
The Rust benchmarks reveal the impact of shared state, locking mechanisms, and optimizations:

1. **Shared State - Tokio RwLock**:
   - **CPU Utilization**: Achieved maximum core utilization (836%) due to efficient handling of concurrent read access.
   - **Memory Usage**: Modest at ~140 MB.
   - **Performance**: Delivered 13.49 requests per second with an average response time of 0.74 seconds.
   - **Observations**: A well-balanced approach but suffered from slight delays.

2. **Shared State - Tokio RwLock (No Padding/Truncate)**:
   - **CPU Utilization**: Slightly higher (876.5%) but led to significant performance degradation.
   - **Memory Usage**: Increased to ~235 MB, likely due to handling larger sequences without truncation.
   - **Performance**: Average response time spiked to 5.81 seconds, resulting in only 497 successful requests.
   - **Observations**: Applying padding and truncation caused inefficiencies.

3. **Non-Shared State - Tokio RwLock**:
   - **CPU Utilization**: Utilized full CPU cores effectively (881.7%).
   - **Memory Usage**: High at ~793 MB due to multiple independent model instances.
   - **Performance**: Comparable to shared state (13.00 requests per second), but with increased memory overhead.
   - **Observations**: Independent model instances removed locking overhead but at a cost to resource efficiency.

4. **Shared State - Mutex**:
   - **CPU Utilization**: Lower (593.6%) due to contention on the Mutex.
   - **Memory Usage**: Lightweight at ~129 MB.
   - **Performance**: Slower at 7.78 requests per second, with an average response time of 1.28 seconds.
   - **Observations**: Mutex contention bottlenecked parallel processing.

5. **Non-Shared State - Mutex**:
   - **CPU Utilization**: High (861.3%), reflecting parallel execution.
   - **Memory Usage**: Similar to the non-shared RwLock configuration (~781 MB).
   - **Performance**: Best among Mutex-based configurations, matching RwLock non-shared performance.
   - **Observations**: Reduced contention improved throughput but maintained high memory costs.

#### **Python Implementation**
The Python implementation, built using FastAPI and Gunicorn, exhibited superior performance under all configurations:

- **CPU Utilization**: Effectively handled concurrency with 10 workers, leveraging preloading for efficiency.
- **Memory Usage**: Moderate at 341.688 MB, balancing resource usage and performance.
- **Performance**: Delivered 278.13 requests per second with an average response time of 0.03 seconds.
- **Observations**: Python's high-level abstractions and optimized libraries like SentenceTransformers excel in handling concurrent requests with minimal overhead.

---

### **Comparative Analysis**

| Metric                  | Rust (Best Config: RwLock Shared) | Python (FastAPI + Gunicorn) |
|--------------------------|------------------------------------|-----------------------------|
| **Average Response Time** | 0.74 seconds                     | 0.03 seconds               |
| **Requests per Second**   | 13.49                             | 278.13                     |
| **Memory Usage**          | 140.9 MB                         | 341.688 MB                 |
| **Concurrency Handling**  | Moderate (lock contention)        | High (preloaded workers)   |
| **Ease of Deployment**    | Moderate (complex dependencies)   | Easy (Python ecosystem)    |

#### **Key Takeaways**
1. **Performance**: Python vastly outperformed Rust in throughput and latency. FastAPI's async handling and preloaded worker processes allowed it to scale efficiently. Moreover, `sentence-transformers` is probably significantly faster than `candle`.
   
2. **Resource Efficiency**: Rust consumed less memory in shared-state configurations but struggled with concurrent read-write scenarios, especially when using Mutex.

3. **Scalability**: Python's Gunicorn worker model is appears to be more scalable for high concurrency workloads compared to Rust.

---

## Benchmarking Text Embedding Inference (TEI)

I was so surprised by the results that I assumed that my Rust implementation had to be too naive, so a more cleaver implementation had to be tested as well. For that I relied on Text Embedding Inference (TEI) from Hugging Face.

The Text Embedding Inference (TEI) library from Hugging Face, implemented in Rust, demonstrates an impressive leap in performance compared to both your earlier Rust implementations and Python benchmarks. Here's a breakdown:

### **TEI Benchmark Analysis**

#### **Observations**
1. **Memory Efficiency**: 
   - Consumes between 123 MB and 193 MB of memory, slightly higher than the most efficient Rust configurations, but still very lightweight for a high-performance application.
   - Maintains efficient memory usage even with increased concurrency levels (100–1000).

2. **Performance Metrics**:
   - **Concurrency Level: 10**
     - **Requests per Second**: 605.51
     - **Average Response Time**: 0.01 seconds
     - **Max Response Time**: 0.26 seconds
   - **Concurrency Level: 100/1000**
     - **Requests per Second**: 1260.26
     - **Average Response Time**: 0.06 seconds
     - **Max Response Time**: 0.12 seconds
   - Scales almost linearly with higher concurrency levels, showcasing exceptional efficiency.

3. **Success Rate**:
   - 100% successful requests across all configurations, indicating robust handling of concurrent requests.

4. **Latency**:
   - Minimal across all concurrency levels, with a 95th percentile response time of 0.02–0.09 seconds.
   - Outperforms both Python and prior Rust implementations by orders of magnitude.

#### **Comparison with Previous Benchmarks**

| Metric                  | TEI (Rust)       | Best Rust Config (RwLock) | Python (FastAPI) |
|--------------------------|------------------|---------------------------|------------------|
| **Memory Usage**         | 123–193 MB       | 140.9 MB                  | 341.688 MB       |
| **Average Response Time**| 0.01–0.06 sec    | 0.74 sec                  | 0.03 sec         |
| **Requests per Second**  | 605.51–1260.26   | 13.49                     | 278.13           |
| **Concurrency Handling** | Excellent        | Moderate                  | Excellent        |

---

### **Key Takeaways**

1. **Rust Optimization at Its Best**:
   - TEI leverages Rust's strengths, including zero-cost abstractions, low-level control, and efficient concurrency, to achieve amazing performance.
   - Likely incorporates advanced optimizations such as **lock-free data structures**, **fine-grained parallelism**, and other strategies.

2. **Comparison with Python**:
   - TEI demonstrates that a Rust implementation, when optimized for a specific use case, can outperform Python libraries like FastAPI both in throughput and latency, even at high concurrency levels.

3. **Scalability**:
   - TEI shows near-linear scalability, maintaining low latencies even under extremely high concurrency (1000 simultaneous requests).

---

### **Final Thoughts**
TEI highlights Rust's potential as a high-performance framework for ML inference tasks, especially in scenarios demanding:
   - Low latency
   - High throughput
   - Memory efficiency
   - Scalable concurrency

For production scenarios, if you can leverage TEI’s performance and adapt it to your use case, it could set a new benchmark for serving ML models. This performance positions Rust as a top-tier choice for inference workloads where resource efficiency and speed are critical.

## Conclusion

Both implementations serve the same purpose but cater to different priorities. Rust excels in (usually) performance and safety, making it ideal for high-stakes production systems. Python, on the other hand, shines in flexibility and ease of use, making it the go-to choice for experimentation and fast development. The right choice depends on your specific use case, performance requirements, and team expertise.

While Rust offers a compelling case for resource-constrained environments, Python remains the superior choice for serving ML models due to its simplicity, ecosystem, and performance in high-concurrency scenarios. Rust implementations could improve with more fine-tuned concurrency strategies, such as actor-based models or sharded state management. In addition, Rust can also enable serverless ML model serving, given its super fast load times and small footprint.

## References

- [Text Embedding Inference](https://github.com/huggingface/text-embeddings-inference)
- [Candle](https://github.com/huggingface/candle)
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [FastAPI](https://fastapi.tiangolo.com)
- [Actix Web](https://actix.rs)