import cv2
import torch
import numpy as np
from PIL import Image
import mediapipe as mp
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, DiffusionPipeline

from typing import Tuple, Union


class BeardMaskGenerator:
    """Generates masks for beard region using facial detection.

    Uses MediaPipe face detection to create a mask covering the lower half
    of detected faces, suitable for beard generation/inpainting.
    """

    def __init__(self, min_detection_confidence: float = 0.5, padding: int = 45):
        """
        Args:
            min_detection_confidence: Minimum confidence for face detection
            padding: Size of dilation kernel for mask padding
        """
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=min_detection_confidence
        )
        self.dilation_kernel = np.ones((padding, padding), np.uint8)

    def get_mask(self, clean_image: Image.Image) -> Image.Image:
        """Generates a binary mask for the beard region.

        Creates a half-ellipse mask on the lower part of each detected face,
        dilated by the padding amount specified in initialization.

        Args:
            clean_image: Input image to generate mask for

        Returns:
            Binary mask image where 255 indicates beard region
        """
        image = np.array(clean_image)
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        results = self.face_detection.process(image)
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                x = max(0, int(bbox.xmin * width))
                y = max(0, int(bbox.ymin * height))
                w = min(int(bbox.width * width), width - x)
                h = min(int(bbox.height * height), height - y)

                center = (x + w // 2, y + h // 2)
                axes = (w // 2, h // 2)
                cv2.ellipse(
                    mask, center, axes, 0, 0, 180, 255, -1
                )  # half circle on bottom part of the face

        mask = cv2.dilate(mask, self.dilation_kernel, iterations=1)  # padding
        mask[mask > 0] = 255
        return Image.fromarray(mask)


def setup_pipelines(
    base_model_str: str = "runwayml/stable-diffusion-v1-5",
    inpaint_model_str: str = "stabilityai/stable-diffusion-2-inpainting",
    device: Union[torch.device, str] = "cuda",
) -> Tuple[DiffusionPipeline, DiffusionPipeline]:
    """Sets up Stable Diffusion pipelines for generation and inpainting.

    Args:
        base_model_str: HuggingFace model ID for base generation
        inpaint_model_str: HuggingFace model ID for inpainting
        device: Device to load models on

    Returns:
        Tuple of (base_pipeline, inpaint_pipeline)
    """
    base_pipe = StableDiffusionPipeline.from_pretrained(
        base_model_str, torch_dtype=torch.float16
    ).to(device)
    base_pipe.set_progress_bar_config(disable=True)
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        inpaint_model_str, torch_dtype=torch.float16
    ).to(device)
    inpaint_pipe.set_progress_bar_config(disable=True)
    return base_pipe, inpaint_pipe
