---
title:  "Image Embedding Inference - Serving Open Source Image Embeddings Model"
mathjax: true
layout: post
categories: media
---

Image embedding is a powerful tool in machine learning that allows us to convert images into numerical vectors, making it easier to analyze, compare, and use them in downstream tasks like classification, clustering, or recommendation systems. This post goes over the [`Image Embedding Inference (IEI)`](https://github.com/bernardo-sb/image-embedding-inference) project, which provides a REST API for generating embeddings using pretrained models from Hugging Face. This project is part of a larger one I'm building which focuses on image retrieval and multi-modal knowledge building.

---


## Overview of Image Embedding Inference (IEI)

The [IEI](https://github.com/bernardo-sb/image-embedding-inference) project offers a REST API for generating embeddings from images using a variety of pretrained models. Its structure and data contracts are inspired by [text-embedding-inference](https://github.com/huggingface/text-embedding-inference). Users can select from a range of models provided by Hugging Face, making it versatile for different use cases.

I started this project because I couldn't find a simple way to serve image embedding models. I wanted a framework as straightforward as the TEI setup for text embeddings, and this project fills that gap.

### Key Features

- **Model Flexibility**: Support for multiple pretrained models, including:
  - `google/vit-base-patch16-224`
  - `facebook/deit-base-patch16-224`
  - `microsoft/beit-base-patch16-224`
  - ...
- **Simple API Design**: A `POST /embed` endpoint accepts base64-encoded images and returns embeddings as numerical vectors.
- **Customizable Model Selection**: Specify the model using the `IMAGE_EMBEDDING_MODEL` environment variable.

---

## API Usage - Example Request

Send a POST request to the `/embed` endpoint with a list of base64-encoded images:

```bash
curl -X POST 'http://localhost:8000/embed' \
--header 'Content-Type: application/json' \
-d '{
    "inputs": ["iVBORw0KGgoAAAANSUhEUgAAAogAAAQwCAYAAABmAK+YAAAMSWlDQ1BJQ0MgUHJvZm..."]
}'
```

### Example Response
The API returns a list of embeddings for each image:

```json
[
    [0.8418501019477844, 0.09062539786100388, 0.21319620311260223, ...],
    ...
]
```

---

## Code Walkthrough

### `server.py`

The main server script is built using FastAPI.

#### 1. **Model Management**

The `ModelManager` class encapsulates model loading and processing logic:

```python
class ModelManager:
    MODELS = [
        "google/vit-base-patch16-224",
        "facebook/deit-base-patch16-224",
        ...
    ]

    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = AutoModel.from_pretrained(model_name, add_pooling_layer=False).to(self.device)
            self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

    @classmethod
    def is_valid_model_name(cls, model_name: str) -> bool:
        return model_name in cls.MODELS

    @classmethod
    def from_model_name(cls, model_name: str) -> "ModelManager":
        if not cls.is_valid_model_name(model_name):
            raise ValueError(f"Invalid model name: {model_name}")
        return cls(model_name)
```

- **Device Selection**: Automatically detects GPU availability for faster inference.
- **Model Loading**: Loads the specified Hugging Face model and its corresponding processor.

#### 2. **Request Validation**

The `EmbeddingRequest` class defines the expected input format:

```python
class EmbeddingRequest(BaseModel):
    inputs: list[str]  # base64 encoded images
```

This ensures that incoming requests contain a valid list of base64-encoded strings.

#### 3. **Image Decoding**

The `decode_base64_image` function converts base64 strings into PIL images (required by the model processor) :

```python
def decode_base64_image(base64_string: str) -> Image.Image:
    if not base64_string:
        raise ValueError("Empty image data")
        
    try:
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")
```

- **Error Handling**: Raises exceptions for invalid or empty image data.

#### 4. **Embedding Generation**

The core logic resides in the `embed` endpoint:

```python
@router.post("/embed", response_model=list[list[float]])
async def embed(
    request: EmbeddingRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    try:
        print(f"Embedding images {len(request.inputs)}...")
        now = datetime.now()
        all_embeddings = []
        for image_input in request.inputs:
            image = decode_base64_image(image_input)
            image_array = np.array(image)
            
            if image_array.ndim != 3:
                raise ValueError("Image must be RGB or RGBA")
            
            if image_array.shape[2] == 4:
                image_array = image_array[:, :, :3]
            
            # Get required input size from processor
            input_size = model_manager.processor.size
            image = Image.fromarray(image_array).resize((input_size["width"], input_size["height"]))
            
            inputs = model_manager.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(model_manager.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model_manager.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()[0]
            
            all_embeddings.append(embeddings)

        total_time = (datetime.now() - now).total_seconds()
        print(f"Embeddings generated in {total_time} seconds. | Avg time per image: {total_time / len(request.inputs)} seconds.")
        return all_embeddings
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")
```

- **Validate Images Shape**: Make sure that images have the right shape and that channels are the last dimension.
- **Preprocessing:** Images are resized to match the model's expected input size.
- **Batch Processing**: Handles multiple images in a single request.
- **Embeddings Extraction**: Computes the mean of the last hidden state to generate a fixed-size vector.

#### 5. **Health Check Endpoint**

A simple `/health` endpoint verifies server status:

```python
@router.get("/health")
async def health():
    return {"message": "OK"}
```

---

## Final Thoughts

The Image Embedding Inference project demonstrates how to create a scalable API for generating image embeddings using Hugging Face models. Its modular design, powered by FastAPI and PyTorch, ensures flexibility and performance.

This API is ideal for applications like visual search, similarity detection, and clustering. Try it out by following the setup instructions on the [repo](https://github.com/bernardo-sb/image-embedding-inference), and experiment with different models to suit your needs!

**Future Improvements:**

- Add proper logging
- Async: image encoding/decoding and model run
- Model information endpoint
- Batch endpoint for running large batch in the background

## References
- [GitHub Repository - Image Embedding Inference](https://github.com/bernardo-sb/image-embeddings-inference)
- [Text Embedding Inference](https://github.com/huggingface/text-embeddings-inference)