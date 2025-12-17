import io
import os
import torch
import logging
import numpy as np
from PIL import Image
from typing import Optional
from pydantic import BaseModel
from diffusers import FluxPipeline
from fastapi.responses import Response
from fastapi import FastAPI, HTTPException, Depends, Header

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global variable to store the pipeline
pipe = None

class ImageRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None
    guidance_scale: float = 3.5
    num_inference_steps: int = 20
    max_sequence_length: int = 256

# Add token verification
def verify_enter_token(x_enter_token: Optional[str] = Header(None)):
    expected_token = os.getenv("ENTER_TOKEN")
    if expected_token and x_enter_token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid or missing token")
    return True

def load_model():
    """Load the Flux model pipeline"""
    global pipe
    try:
        logger.info("Starting model load...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16
        )
        logger.info("Model loaded, moving to CUDA...")
        pipe.to("cuda")
        logger.info("Model ready on CUDA")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": pipe is not None
    }

def calculate_generation_dimensions(target_w: int, target_h: int):
    """
    Calculate optimal generation dimensions and whether upscaling is needed.
    
    Args:
        target_w: Target width requested by user
        target_h: Target height requested by user
    
    Returns:
        tuple: (generation_width, generation_height, final_width, final_height, should_upscale)
    """
    # Maximum dimension for direct generation
    MAX_DIRECT_DIM = 1024
    # Minimum dimension to avoid quality issues
    MIN_DIM = 256
    # Tile size for upscaling
    TILE_SIZE = 1024
    
    # If both dimensions are at or below MAX_DIRECT_DIM, generate directly at target size
    if target_w <= MAX_DIRECT_DIM and target_h <= MAX_DIRECT_DIM:
        # Ensure dimensions are multiples of 8 (required by model)
        gen_w = (target_w // 8) * 8
        gen_h = (target_h // 8) * 8
        return gen_w, gen_h, gen_w, gen_h, False
    
    # For larger images, we need to generate at a lower resolution and upscale
    aspect_ratio = target_w / target_h
    
    # Calculate generation dimensions that maintain aspect ratio
    # and are close to MAX_DIRECT_DIM
    if aspect_ratio > 1:
        # Landscape
        gen_w = MAX_DIRECT_DIM
        gen_h = int(MAX_DIRECT_DIM / aspect_ratio)
    else:
        # Portrait or square
        gen_h = MAX_DIRECT_DIM
        gen_w = int(MAX_DIRECT_DIM * aspect_ratio)
    
    # Ensure generation dimensions are multiples of 8 and above minimum
    gen_w = max(MIN_DIM, (gen_w // 8) * 8)
    gen_h = max(MIN_DIM, (gen_h // 8) * 8)
    
    # Final dimensions should match the request
    final_w = target_w
    final_h = target_h
    
    return gen_w, gen_h, final_w, final_h, True

def upscale_image(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """
    Upscale an image to target dimensions using high-quality resampling.
    
    Args:
        image: PIL Image to upscale
        target_width: Target width
        target_height: Target height
    
    Returns:
        Upscaled PIL Image
    """
    logger.info(f"Upscaling from {image.size} to {target_width}x{target_height}")
    
    # Use Lanczos resampling for high quality
    upscaled = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    return upscaled

def generate_image(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    guidance_scale: float,
    num_inference_steps: int,
    max_sequence_length: int,
    generator: torch.Generator
) -> Image.Image:
    """
    Generate an image using the Flux pipeline.
    
    Args:
        prompt: Text prompt for image generation
        width: Width of the image to generate
        height: Height of the image to generate
        seed: Random seed for reproducibility
        guidance_scale: Guidance scale for generation
        num_inference_steps: Number of inference steps
        max_sequence_length: Maximum sequence length for text encoding
        generator: PyTorch random generator
    
    Returns:
        Generated PIL Image
    """
    logger.info(f"Generating image: {width}x{height}, prompt: '{prompt[:50]}...'")
    
    try:
        result = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            generator=generator
        )
        
        if result.images and len(result.images) > 0:
            return result.images[0]
        else:
            raise ValueError("No image generated")
            
    except Exception as e:
        logger.error(f"Error during image generation: {str(e)}")
        raise

def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """
    Convert a PIL Image to bytes.
    
    Args:
        image: PIL Image to convert
        format: Output format (PNG, JPEG, etc.)
    
    Returns:
        Image as bytes
    """
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()

def resize_to_multiple_of_8(width: int, height: int) -> tuple:
    """
    Ensure dimensions are multiples of 8.
    
    Args:
        width: Requested width
        height: Requested height
    
    Returns:
        tuple: (adjusted_width, adjusted_height)
    """
    return (width // 8) * 8, (height // 8) * 8

@app.post("/")
def root_generate(request: ImageRequest, _auth: bool = Depends(verify_enter_token)):
    """Root endpoint that accepts POST requests"""
    return generate(request, _auth)

@app.post("/generate")
def generate(request: ImageRequest, _auth: bool = Depends(verify_enter_token)):
    logger.info(f"Request: {request}")
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Use 1024 as default only if both dimensions are default
    # If one is specified, scale the other to maintain aspect ratio
    width = request.width
    height = request.height
    if width == 1024 and height == 1024:
        # Both are defaults - keep them
        pass
    elif width == 1024 and height != 1024:
        # Only height was specified - scale width proportionally (assume square)
        width = height
    elif height == 1024 and width != 1024:
        # Only width was specified - scale height proportionally (assume square)
        height = width
    
    seed = request.seed if request.seed is not None else int.from_bytes(os.urandom(8), "big")
    logger.info(f"Using seed: {seed}")
    generator = torch.Generator("cuda").manual_seed(seed)
    gen_w, gen_h, final_w, final_h, should_upscale = calculate_generation_dimensions(width, height)
    logger.info(f"Requested: {width}x{height} -> Generation: {gen_w}x{gen_h} -> Final: {final_w}x{final_h} (upscale: {should_upscale})")
    
    try:
        # Generate image at calculated generation dimensions
        image = generate_image(
            prompt=request.prompt,
            width=gen_w,
            height=gen_h,
            seed=seed,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            max_sequence_length=request.max_sequence_length,
            generator=generator
        )
        
        # Upscale if needed
        if should_upscale:
            image = upscale_image(image, final_w, final_h)
        
        # Convert to bytes and return
        image_bytes = image_to_bytes(image)
        
        return Response(
            content=image_bytes,
            media_type="image/png",
            headers={
                "X-Seed": str(seed),
                "X-Generation-Dimensions": f"{gen_w}x{gen_h}",
                "X-Final-Dimensions": f"{final_w}x{final_h}",
                "X-Upscaled": str(should_upscale)
            }
        )
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
