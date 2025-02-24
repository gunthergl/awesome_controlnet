# %%
import random  # RNG

import cv2  # OpenCV for image processing operations
import einops  # Tensor manipulation
import numpy as np  # Numerical operations on arrays
import torch  # PyTorch
from pytorch_lightning import (  # Set seeds for reproducibility in the whole training process
    seed_everything,
)

import cc_pipeline.config  # Configuration settings (e.g., for memory saving)
from cc_pipeline.annotator.util import (  # Utility functions for image processing
    HWC3,
    resize_image,
)
from cc_pipeline.share import *

# resize_image: Resize an image to a given resolution while maintaining aspect ratio
# HWC3: Ensure an image has 3 channels (RGB) and is in Height-Width-Channel format


def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to the LAB color space.

    LAB: L* for lightness (0-100), a* for green-red (-127 to 127), b* for blue-yellow (-127 to 127).

    LAB mimics human vision with consistent color differences and is device-independent,
    making it ideal for tasks like color correction, contrast enhancement, and clustering.

    https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_lab

    Args:
        rgb: Input image in RGB format.

    Returns:
        Image converted to LAB color space (H, W, 3):
            L = [:, :, 0]
            a = [:, :, 1]
            b = [:, :, 2]
    """
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)


def lab2rgb(lab: np.ndarray) -> np.ndarray:
    """
    Convert a LAB image to the RGB color space.

    RGB: Red, Green, Blue color channels (0-255).

    Args:
        lab: Input image in LAB format.

    Returns:
        Image converted to RGB color space (H, W, 3):
            R = [:, :, 0]
            G = [:, :, 1]
            B = [:, :, 2]
    """
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def rgb2yuv(rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to the YUV color space.

    8bit YUV: Y for luminance (0-255), U for chrominance blue (0-255), V for chrominance red  (0-255).

    YUV: compression because it separating brightness (Y) from color (U, V) --> reduce data size while preserving visual quality.

    Args:
        rgb: Input image in RGB format.

    Returns:
        Image converted to YUV color space (H, W, 3):
            Y = [:, :, 0]
            U = [:, :, 1]
            V = [:, :, 2]
    """
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)


def yuv2rgb(yuv: np.ndarray) -> np.ndarray:
    """
    Convert a YUV image to the RGB color space.

    Args:
        yuv: Input image in YUV format.

    Returns:
        Image converted to RGB color space.
    """
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)


def srgb2lin(s):
    """
    Convert an sRGB image to linear RGB.

    sRGB: Standard RGB color space with a non-linear gamma curve (display devices)
    Linear RGB: A version of RGB where values are directly proportional to light intensity.

    Relevant for image processing tasks such as blending, filtering, and physically-based rendering.
    Ensures maths is on correctly scaled light values.

    Args:
        s: sRGB image data.

    Returns:
        Linearized RGB image data.
    """
    s = s.astype(float) / 255.0  # Normalize sRGB values to [0,1]
    # Definition from https://en.wikipedia.org/wiki/SRGB "transfer function gamma"
    return np.where(
        s <= 0.0404482362771082, s / 12.92, np.power(((s + 0.055) / 1.055), 2.4)
    )


def lin2srgb(lin):
    """
    Convert a linear RGB image back to sRGB.

    Args:
        lin: Linear RGB image data.

    Returns:
        sRGB image data scaled back to [0,255].
    """
    return 255 * np.where(
        lin > 0.0031308, 1.055 * (np.power(lin, (1.0 / 2.4))) - 0.055, 12.92 * lin
    )


def get_luminance(
    linear_image: np.ndarray, luminance_conversion=[0.2126, 0.7152, 0.0722]
):
    """
    Calculate the luminance of an image given its linear RGB values.

    Args:
        linear_image: The image in linear RGB space.
        luminance_conversion: Conversion factors for RGB channels.

    Returns:
        The computed luminance of the image.
    """
    # Multiply the linear image by the conversion factors and sum across channels.
    return np.sum([[luminance_conversion]] * linear_image, axis=2)


def take_luminance_from_first_chroma_from_second(luminance, chroma, mode="lab", s=1):
    """
    Replace the luminance channel of the chroma image with that from the luminance image.

    Why?: Given grayscale is in good detail, and color is not, we can use the grayscale image's luminance
    and superimpose color from the chroma image to create a more detailed color image.

    Args:
        luminance: The image providing the luminance channel.
        chroma: The image providing the chroma (color) channels.
        mode: The color space to use ('lab', 'yuv', or 'luminance').
        s: Exponent factor used in luminance mode adjustment.

    Returns:
        A new image combining the luminance from the first image with the chroma from the second.
    """
    # Ensure both images have the same shape.
    assert luminance.shape == chroma.shape, f"{luminance.shape=} != {chroma.shape=}"

    if mode == "lab":
        # Convert chroma image to LAB color space
        lab = rgb2lab(chroma)
        # Replace the L channel with the L channel from the luminance image
        lab[:, :, 0] = rgb2lab(luminance)[:, :, 0]
        # Convert back to RGB
        return lab2rgb(lab)

    if mode == "yuv":
        # Convert chroma image to YUV color space
        yuv = rgb2yuv(chroma)
        # Replace the Y (luminance) channel with the one from the luminance image
        yuv[:, :, 0] = rgb2yuv(luminance)[:, :, 0]
        # Convert back to RGB
        return yuv2rgb(yuv)

    if mode == "luminance":
        # Convert images to linear RGB
        lluminance = srgb2lin(luminance)
        lchroma = srgb2lin(chroma)
        # Adjust chroma using the ratio of luminances and convert back to sRGB
        return lin2srgb(
            np.clip(
                lchroma
                * ((get_luminance(lluminance) / (get_luminance(lchroma))) ** s)[
                    :, :, np.newaxis
                ],
                0,
                1,
            )
        )


# %%
# =============================================================================
# Utility Functions for Image Processing and Diffusion Sampling
# =============================================================================


def process(
    input_image,
    prompt,
    a_prompt,
    n_prompt,
    num_samples,
    image_resolution,
    ddim_steps,
    guess_mode,
    strength,
    scale,
    seed,
    eta,
    low_threshold,
    high_threshold,
    ddim_sampler,
    apply_canny,
    model,
):
    """
    Process the input image using Canny edge detection and a diffusion model sampler.

    Args:
        input_image: The input image array.
        prompt: The text prompt guiding the synthesis.
        a_prompt: Additional positive prompt details.
        n_prompt: Negative prompt details.
        num_samples: Number of samples to generate.
        image_resolution: The resolution to which the image will be resized.
        ddim_steps: Number of DDIM sampling steps.
        guess_mode: Boolean flag to determine how control scales are set.
        strength: Strength multiplier for the control signal.
        scale: Guidance scale for conditioning.
        seed: Seed for random number generation; if -1, a random seed is chosen.
        eta: Parameter controlling stochasticity in the sampler.
        low_threshold: Low threshold for Canny edge detection.
        high_threshold: High threshold for Canny edge detection.

    Returns:
        A list of images: the inverse of the detected edge map and the generated samples.
    """
    with torch.no_grad():
        # Resize the input image to the desired resolution
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        # Apply Canny edge detection to the image.
        detected_map = apply_canny(img, low_threshold, high_threshold)
        # Ensure the detected map has 3 channels.
        detected_map = HWC3(detected_map)

        # Convert the detected map to a torch tensor, normalize, and prepare for model input.
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        # Duplicate the control map for each of num_samples.
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        # Rearrange tensor dimensions to match the model's expected input (batch, channels, height, width).
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        # If seed is -1, pick a random seed.
        if seed == -1:
            seed = random.randint(0, 65535)
        # Set the random seed for reproducibility.
        seed_everything(seed)

        # Optionally reduce memory usage if enabled in the config.
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        # Prepare the conditioning dictionary for the diffusion model.
        cond = {
            "c_concat": [control],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
            ],
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
        }
        # Define the shape for the latent space (channels, height, width).
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        # Set control scales based on the guess mode.
        # When in guess_mode, scales decay exponentially; otherwise, they are constant.
        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode
            else ([strength] * 13)
        )  # Magic number: 0.825**12 < 0.01 but 0.826**12 > 0.01

        # Run the DDIM sampling process.
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        # Decode latent space into full-resolution space
        x_samples = model.decode_first_stage(samples)
        # Rearrange tensor dimensions back to (batch, height, width, channels) and scale pixel values.
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        # Collect the generated samples into a list.
        results = [x_samples[i] for i in range(num_samples)]

    # Return the inverted detected edge map along with the generated images.
    return [255 - detected_map] + results
