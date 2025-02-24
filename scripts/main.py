"""
MRI Image Generator API

This FastAPI application provides an endpoint to generate MRI-like images using a diffusion model with Canny edge detection.
It loads a pre-trained model and processes input images with specified parameters.

Modules Used:
- FastAPI: To create the API.
- PIL & imageio: To handle image processing.
- NumPy: For numerical operations.
- Matplotlib: To visualize the results.
- Pydantic: To define request validation.
- pathlib, io, os: For file handling.
- Custom modules: Image processing utilities and model handling.

Usage:
Run the API using the command:
    uvicorn main:app --host localhost --port 5001 --reload

Example Requests:
1. Generate an image with default settings:
   curl -X 'POST' 'http://localhost:5001/generate' --output output_image.png
2. Generate an image with custom parameters:
   curl -X 'POST' 'http://localhost:5001/generate' -H 'Content-Type: application/json' -d '{"strength": 0.7,"low_threshold": 30}' -o removeme.png

"""

import io
from contextlib import asynccontextmanager

import imageio
import matplotlib
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from cc_pipeline.annotator.canny import (  # Canny edge detection implementation
    CannyDetector,
)
from cc_pipeline.annotator.util import HWC3  # Utility functions for image processing

# Import custom image processing and model handling functions
from cc_pipeline.awesomedemo_fastapi import (
    process,
    resize_image,
    take_luminance_from_first_chroma_from_second,
)
from cc_pipeline.cldm.ddim_hacked import (  # Sampler for DDIM diffusion process
    DDIMSampler,
)
from cc_pipeline.cldm.model import (  # Functions to create and load models
    create_model,
    load_state_dict,
)

matplotlib.use("AGG")
import matplotlib.pyplot as plt


class ImageRequest(BaseModel):
    """
    Defines the structure of the request for image generation.
    """

    prompt: str = "mri brain scan"
    num_samples: int = 1
    image_resolution: int = 128  # Resolution of the output image (default 128px)
    strength: float = 1.0  # Strength of the diffusion model effect
    guess_mode: bool = False  # Whether to enable guess mode
    low_threshold: int = 50  # Lower threshold for edge detection
    high_threshold: int = 100  # Upper threshold for edge detection
    ddim_steps: int = 4  # Number of steps in the DDIM sampling process
    scale: float = 9.0  # Scaling factor for image processing
    seed: int = 1  # Random seed for reproducibility
    eta: float = 0.0  # Noise parameter for DDIM sampler
    a_prompt: str = "good quality"  # Positive prompt for generation
    n_prompt: str = "animal, drawing, painting, vivid colors, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"  # Negative prompt to filter unwanted results


# Global variables for model and sampler
model = None
ddim_sampler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle event to initialize the model and sampler when the app starts.
    """
    global model, ddim_sampler
    model, ddim_sampler = setup_model_sampler()
    print("Model and DDIM Sampler Initialized")
    yield  # Allow app to continue running


app = FastAPI(lifespan=lifespan)


def setup_model_sampler():
    """
    Initializes and loads the pre-trained model and DDIM sampler.
    """
    print("SETTING UP")
    global model, ddim_sampler

    # Load the model configuration and initialize it on the CPU
    model = create_model("./models/cldm_v15.yaml").cpu()

    # Load pre-trained weights
    model.load_state_dict(load_state_dict("./models/control_sd15_canny.pth"))

    # Move the model to GPU for faster computation
    model = model.cuda()

    # Denoising Diffusion Implicit Model (DDIM)
    # Initialize the DDIM sampler with the model.
    ddim_sampler = DDIMSampler(model)

    print("DDIM Sampler Initialized")
    return model, ddim_sampler


@app.post("/generate")
async def generate_image(request: ImageRequest = None):
    """
    Endpoint to generate an MRI-like image using the pre-trained model and input parameters.

    Parameters:
    - request (ImageRequest, optional): Input parameters for image generation.

    Returns:
    - StreamingResponse: A PNG image generated based on the input parameters.
    """
    global model, ddim_sampler

    if request is None:
        request = ImageRequest()

    # Convert request to dictionary and merge with default values
    request_dict = jsonable_encoder(request)
    default_request = ImageRequest().dict()
    final_request = {
        **default_request,
        **request_dict,
    }  # Merge defaults with provided values

    try:
        # Load input image for processing
        current_input = imageio.imread("test_imgs/mri_brain.jpg")

        # Process the input image using the model and DDIM sampler
        result = process(
            input_image=current_input,
            model=model,
            ddim_sampler=ddim_sampler,
            apply_canny=CannyDetector(),
            **final_request,
        )

        # Apply luminance transformation to the output image
        transformed_image = take_luminance_from_first_chroma_from_second(
            resize_image(HWC3(current_input), request.image_resolution),
            result[-1],
            mode="lab",
        )

        # Save the output image as a PNG and return it as a response
        img_buf = io.BytesIO()
        plt.imshow(transformed_image)
        plt.axis("off")
        plt.savefig(img_buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close()
        img_buf.seek(0)

        return StreamingResponse(img_buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def home():
    """
    Root endpoint to check API status.

    Returns:
    - dict: A message confirming API is running.
    """
    return {"message": "MRI Image Generator API"}


if __name__ == "__main__":
    """
    Run the FastAPI application with Uvicorn.
    """
    uvicorn.run("main:app", host="localhost", port=5001, reload=True)
