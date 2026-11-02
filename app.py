from __future__ import annotations

import tempfile
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Final, cast

import albumentations as A
import av  # type: ignore[import-untyped]
import gradio as gr
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from beartype import beartype
from jaxtyping import Float, UInt8, jaxtyped
from PIL import Image
from torch import Tensor, nn

IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset({".jpg"})
VIDEO_SUFFIXES: Final[frozenset[str]] = frozenset({".mp4", ".mov", ".mkv"})
MODEL_SUFFIX: Final[str] = ".pt"
MODEL_DIR: Final[Path] = Path("./model")
MASK_ALPHA: Final[float] = 0.5
DEFAULT_VIDEO_RATE: Final[Fraction] = Fraction(30, 1)
IMAGE_HEIGHT: Final[int] = 360
IMAGE_WIDTH: Final[int] = 640
NUM_CLASSES: Final[int] = 20

# Keep the palette identical to evaluate_performance.py so predictions shown in
# the Gradio app use the same semantic colours as the project's evaluation
# visualisations.
CLASS_COLORS: Final[np.ndarray] = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [0, 0, 0],
    ],
    dtype=np.uint8,
)

# These aliases mirror the tensor/array shapes documented in
# evaluate_performance.py and let jaxtyping catch shape mismatches close to the
# source instead of surfacing a confusing error later in the UI callback. Source
# image axes deliberately use different symbolic names from model-space axes:
# preprocessing resizes arbitrary source dimensions to 360x640, so reusing the
# same ``h``/``w`` symbols would incorrectly require both shapes to be identical.
NumpyImage = Float[np.ndarray, "source_h source_w 3"]
BatchImage = Float[Tensor, "b 3 model_h model_w"]
Logits = Float[Tensor, "b 20 model_h model_w"]
ClassIndexArray = UInt8[np.ndarray, "mask_h mask_w"]
ColorMask = UInt8[np.ndarray, "mask_h mask_w 3"]

# Match the evaluation pipeline: every inference input is resized to 360x640,
# while ToTensorV2 converts the HWC float array to CHW without introducing an
# extra normalization step.
INFERENCE_TRANSFORM: Final[A.Compose] = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        ToTensorV2(),
    ]
)

APP_CSS: Final[str] = """
.gradio-container {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 24px 3vw !important;
}
#app-title {
    text-align: center;
    margin: 12px 0 6px 0;
}
#model-selector {
    width: min(440px, 90vw) !important;
    max-width: 440px !important;
    margin: 0 auto 28px auto;
}
#workspace {
    width: min(1500px, 94vw) !important;
    max-width: 1500px !important;
    margin: 0 auto !important;
    border: 2px solid var(--border-color-primary);
    border-radius: 24px;
    padding: 18px;
}
#workspace > div {
    width: 100% !important;
}
#workspace .tab-nav {
    width: 100% !important;
}
#workspace .tab-nav button {
    flex: 1 1 50% !important;
    justify-content: center !important;
}
#workspace .tabitem {
    width: 100% !important;
}
.panel-box {
    border: 2px solid var(--border-color-primary) !important;
    border-radius: 24px !important;
    padding: 16px !important;
    min-height: 390px;
}
.panel-heading {
    text-align: center;
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 12px;
}
.result-frame {
    min-height: 300px;
}
"""


def discover_model_checkpoints(model_dir: Path) -> list[Path]:
    """Discover top-level PyTorch checkpoints available to the application.

    The function checks the configured model directory, selects regular files
    whose suffix is ``.pt``, and sorts them by filename so the dropdown order is
    deterministic across launches.

    Parameters
    ----------
    model_dir : pathlib.Path
        Directory containing selectable model checkpoints.

    Returns
    -------
    list[pathlib.Path]
        Sorted top-level ``.pt`` checkpoint paths.

    Raises
    ------
    RuntimeError
        If the model directory does not exist or contains no ``.pt`` files.
    """
    # Fail at startup rather than presenting an empty dropdown that can never
    # complete inference.
    if not model_dir.is_dir():
        raise RuntimeError(f"Model directory does not exist: {model_dir}")

    checkpoints = sorted(
        (
            path
            for path in model_dir.iterdir()
            if path.is_file() and path.suffix.lower() == MODEL_SUFFIX
        ),
        key=lambda path: path.name.lower(),
    )
    if not checkpoints:
        raise RuntimeError(f"No {MODEL_SUFFIX} checkpoints found in {model_dir}")
    return checkpoints


class ModelRepository:
    """Discover and cache the trained segmentation models under ``./model``.

    The repository exposes all top-level ``.pt`` files to the Gradio dropdown.
    A selected checkpoint is deserialized exactly like ``evaluate_performance.py``:
    ``torch.load(..., weights_only=False)``, followed by an ``nn.Module`` type
    check, device transfer, and ``eval()``. Models are loaded lazily and cached
    to avoid repeatedly deserializing a checkpoint during image/video requests.

    Parameters
    ----------
    model_dir : pathlib.Path
        Directory containing trained model checkpoints.
    device : torch.device
        Device on which loaded models will execute.

    Raises
    ------
    RuntimeError
        If ``model_dir`` is missing or contains no top-level ``.pt`` files.
    """

    def __init__(self, model_dir: Path, device: torch.device) -> None:
        self._device = device
        self._checkpoints = discover_model_checkpoints(model_dir)
        self._choice_to_path = {
            self._display_path(path): path for path in self._checkpoints
        }
        self._cache: dict[str, nn.Module] = {}

    @property
    def choices(self) -> list[str]:
        """Return every discovered checkpoint as a dropdown display value.

        Returns
        -------
        list[str]
            Relative checkpoint paths such as ``./model/student_V5.pt``.
        """
        return list(self._choice_to_path)

    @property
    def default_choice(self) -> str:
        """Return the preferred initial checkpoint selection.

        ``student_V5.pt`` is selected when available because it matches the
        supplied UI design. Otherwise the first alphabetically sorted checkpoint
        is used to keep startup deterministic.

        Returns
        -------
        str
            Dropdown value used as the initial model selection.
        """
        preferred = "./model/student_V5.pt"
        if preferred in self._choice_to_path:
            return preferred
        return self.choices[0]

    def get_model(self, checkpoint_choice: str) -> nn.Module:
        """Return a selected model, loading and caching it on first use.

        Steps
        -----
        1. Resolve the dropdown value to a discovered checkpoint path.
        2. Reuse a cached model when the checkpoint has already been loaded.
        3. Deserialize the full ``nn.Module`` with ``weights_only=False``.
        4. Move the model to the execution device and enable evaluation mode.

        Parameters
        ----------
        checkpoint_choice : str
            Dropdown value identifying one discovered checkpoint.

        Returns
        -------
        torch.nn.Module
            Evaluation-mode model placed on the configured device.

        Raises
        ------
        gradio.Error
            If the selection is invalid, the checkpoint cannot be read, or the
            deserialized object is not an ``nn.Module``.
        """
        path = self._choice_to_path.get(checkpoint_choice)
        if path is None:
            raise gr.Error("Select a valid model checkpoint from the dropdown.")

        cached_model = self._cache.get(checkpoint_choice)
        if cached_model is not None:
            return cached_model

        try:
            # This is intentionally the same deserialization call used by
            # evaluate_performance.py because the training scripts save complete
            # nn.Module objects rather than state_dict-only checkpoints.
            checkpoint = torch.load(
                path,
                map_location=self._device,
                weights_only=False,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise gr.Error(f"Could not load checkpoint {path.name}: {exc}") from exc

        if not isinstance(checkpoint, nn.Module):
            raise gr.Error(
                f"Checkpoint {path.name} loaded an object of type "
                f"{type(checkpoint).__name__}, but an nn.Module was expected."
            )

        model = checkpoint.to(self._device)
        model.eval()
        self._cache[checkpoint_choice] = model
        return model

    @staticmethod
    def _display_path(path: Path) -> str:
        """Convert a checkpoint path to the relative format shown in the UI.

        Parameters
        ----------
        path : pathlib.Path
            Discovered checkpoint path.

        Returns
        -------
        str
            Path rendered with an explicit ``./`` prefix.
        """
        return f"./{path.as_posix().removeprefix('./')}"


@jaxtyped(typechecker=beartype)
def prepare_image_tensor(image: NumpyImage) -> BatchImage:
    """Prepare one normalized RGB image for model inference.

    This follows the evaluation dataset path: the input is already float RGB in
    ``[0, 1]``, Albumentations resizes it to ``360 x 640``, and ToTensorV2
    converts HWC to CHW. A batch dimension is added before returning.

    Parameters
    ----------
    image : np.ndarray
        Float RGB image shaped ``(H, W, 3)`` with values in ``[0, 1]``.

    Returns
    -------
    torch.Tensor
        Batched float tensor shaped ``(1, 3, 360, 640)``.
    """
    transformed = INFERENCE_TRANSFORM(image=image)
    image_tensor = cast(Tensor, transformed["image"])
    return cast(BatchImage, image_tensor.unsqueeze(0))


@jaxtyped(typechecker=beartype)
def forward(model: nn.Module, image_batch: BatchImage) -> Logits:
    """Run one segmentation forward pass and validate the output shape.

    The helper intentionally mirrors ``evaluate_performance.py``: the model is
    called directly and is expected to return 20-class logits in NCHW layout.

    Parameters
    ----------
    model : torch.nn.Module
        Trained semantic-segmentation model.
    image_batch : torch.Tensor
        Input batch shaped ``(B, 3, H, W)``.

    Returns
    -------
    torch.Tensor
        Raw logits shaped ``(B, 20, H, W)``.

    Raises
    ------
    beartype.roar.BeartypeCallHintReturnViolation
        If the model output does not match the expected tensor shape.
    """
    return cast(Logits, model(image_batch))


@jaxtyped(typechecker=beartype)
def colorize_mask(class_mask: ClassIndexArray) -> ColorMask:
    """Map a class-index prediction to the BDD100k RGB palette.

    Parameters
    ----------
    class_mask : np.ndarray
        Predicted class IDs shaped ``(H, W)`` with values in ``[0, 19]``.

    Returns
    -------
    np.ndarray
        RGB colour mask shaped ``(H, W, 3)`` with dtype ``uint8``.

    Raises
    ------
    IndexError
        If ``class_mask`` contains an index outside the 20-class palette.
    """
    return cast(ColorMask, CLASS_COLORS[class_mask.astype(np.int64)])


class SemanticSegmenter:
    """Run checkpoint inference and render the predicted semantic-mask overlay.

    The service follows the project's evaluation path for preprocessing and
    inference. Each source is converted to RGB float pixels in ``[0, 1]``,
    resized to ``360 x 640`` by the shared inference transform, passed through
    the selected model under ``torch.inference_mode()``, reduced with ``argmax``,
    and colourized with the BDD100k palette. The mask is finally resized back to
    the source dimensions and blended at 50% opacity for display.

    Parameters
    ----------
    repository : ModelRepository
        Repository resolving dropdown choices to trained models.
    device : torch.device
        Device used for input tensors and inference.
    """

    def __init__(self, repository: ModelRepository, device: torch.device) -> None:
        self._repository = repository
        self._device = device

    @torch.inference_mode()
    def segment(self, image: Image.Image, checkpoint_choice: str) -> Image.Image:
        """Segment one image or decoded video frame with a selected checkpoint.

        Steps
        -----
        1. Convert the source to RGB and normalize pixels to ``[0, 1]``.
        2. Apply the same ``360 x 640`` inference transform as evaluation.
        3. Run the selected model and collapse logits with ``argmax``.
        4. Colourize the 20-class prediction with the project palette.
        5. Resize the mask to the source resolution and blend it over the image.

        Parameters
        ----------
        image : PIL.Image.Image
            Source image or one decoded video frame.
        checkpoint_choice : str
            Dropdown value identifying the model checkpoint for inference.

        Returns
        -------
        PIL.Image.Image
            RGB source image with its predicted segmentation mask overlaid.

        Raises
        ------
        gradio.Error
            If model loading or inference fails, including incompatible output
            shapes or class indices.
        """
        source = image.convert("RGB")
        image_array = np.asarray(source, dtype=np.float32) / 255.0

        try:
            image_batch = prepare_image_tensor(cast(NumpyImage, image_array)).to(
                self._device
            )
            model = self._repository.get_model(checkpoint_choice)

            # The evaluator unwraps DataParallel for single-image visualization;
            # doing the same here keeps one-frame inference behavior consistent.
            inference_model = model.module if isinstance(model, nn.DataParallel) else model
            inference_model.eval()
            logits = forward(inference_model, image_batch)
            pred_class = (
                logits.argmax(dim=1)
                .squeeze(0)
                .to(device="cpu", dtype=torch.uint8)
                .numpy()
            )
            pred_color = colorize_mask(cast(ClassIndexArray, pred_class))
        except gr.Error:
            raise
        except (IndexError, RuntimeError, TypeError, ValueError) as exc:
            raise gr.Error(f"Segmentation inference failed: {exc}") from exc

        # Evaluation visualizations blend the predicted color mask at alpha=0.5.
        # Resize with nearest-neighbor so semantic class colours are not mixed at
        # boundaries when restoring the source image's display resolution.
        mask_image = Image.fromarray(pred_color, mode="RGB").resize(
            source.size,
            resample=Image.Resampling.NEAREST,
        )
        return Image.blend(source, mask_image, alpha=MASK_ALPHA)


def _validate_suffix(file_path: str, allowed_suffixes: frozenset[str]) -> Path:
    """Validate an uploaded file path against an explicit suffix allow-list.

    Parameters
    ----------
    file_path : str
        Filesystem path returned by the Gradio file component.
    allowed_suffixes : frozenset[str]
        Lowercase file suffixes accepted for the active tab.

    Returns
    -------
    pathlib.Path
        Validated path object for downstream processing.

    Raises
    ------
    gradio.Error
        If the file extension is not allowed or the uploaded path is missing.
    """
    path = Path(file_path)

    # Browser picker filters improve UX but are not a validation boundary, so
    # enforce the extension again on the server before decoding user content.
    if path.suffix.lower() not in allowed_suffixes:
        allowed = ", ".join(sorted(allowed_suffixes))
        raise gr.Error(f"Unsupported file type. Allowed extensions: {allowed}")
    if not path.is_file():
        raise gr.Error("The uploaded file is no longer available on the server.")
    return path


def segment_image(
    file_path: str,
    checkpoint_choice: str,
    segmenter: SemanticSegmenter,
) -> Image.Image:
    """Validate and segment an uploaded JPG with the selected checkpoint.

    Parameters
    ----------
    file_path : str
        Path to the uploaded JPG supplied by Gradio.
    checkpoint_choice : str
        Dropdown value identifying the model used for inference.
    segmenter : SemanticSegmenter
        Reusable segmentation service shared by the application.

    Returns
    -------
    PIL.Image.Image
        Original image with its semantic segmentation mask blended on top.

    Raises
    ------
    gradio.Error
        If the extension is invalid, Pillow cannot decode the image, or model
        inference fails.
    """
    path = _validate_suffix(file_path, IMAGE_SUFFIXES)
    try:
        # Open within the callback so uploaded-file handles are released as soon
        # as the request completes.
        with Image.open(path) as image:
            return segmenter.segment(image, checkpoint_choice)
    except gr.Error:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise gr.Error(f"Could not process the JPG image: {exc}") from exc


def segment_video(
    file_path: str,
    checkpoint_choice: str,
    segmenter: SemanticSegmenter,
) -> str:
    """Segment every frame of an uploaded video with one selected checkpoint.

    The video is decoded frame-by-frame with PyAV. Every frame goes through the
    exact same segmentation path as a still image, then the overlays are encoded
    to H.264 MP4 for browser playback. Audio is intentionally omitted because
    the requested result is the segmented visual stream.

    Parameters
    ----------
    file_path : str
        Path to an uploaded MP4, MOV, or MKV video supplied by Gradio.
    checkpoint_choice : str
        Dropdown value identifying the model used for every video frame.
    segmenter : SemanticSegmenter
        Reusable segmentation service shared by the application.

    Returns
    -------
    str
        Path to the generated H.264 MP4 displayed by Gradio.

    Raises
    ------
    gradio.Error
        If validation, decoding, segmentation inference, or encoding fails.
    """
    path = _validate_suffix(file_path, VIDEO_SUFFIXES)

    # NamedTemporaryFile safely allocates a unique path and closes the handle
    # before PyAV opens it for writing, avoiding descriptor leaks from mkstemp.
    with tempfile.NamedTemporaryFile(prefix="segmented_", suffix=".mp4", delete=False) as tmp:
        output_path = Path(tmp.name)

    try:
        with av.open(str(path), mode="r") as input_container:
            if not input_container.streams.video:
                raise gr.Error("The uploaded file does not contain a video stream.")

            input_stream = input_container.streams.video[0]
            frame_rate = input_stream.average_rate or DEFAULT_VIDEO_RATE

            with av.open(str(output_path), mode="w") as output_container:
                # H.264/yuv420p is broadly supported by web browsers and Gradio.
                output_stream = output_container.add_stream("libx264", rate=frame_rate)
                output_stream.width = input_stream.codec_context.width
                output_stream.height = input_stream.codec_context.height
                output_stream.pix_fmt = "yuv420p"

                # Resolve/load the checkpoint on the first frame, then the model
                # repository cache keeps every subsequent frame on the same model.
                for frame in input_container.decode(input_stream):
                    segmented = segmenter.segment(frame.to_image(), checkpoint_choice)
                    encoded_frame = av.VideoFrame.from_image(segmented)
                    for packet in output_stream.encode(encoded_frame):
                        output_container.mux(packet)

                # Flush delayed encoder packets so the final MP4 is complete and
                # seekable before Gradio serves it to the browser.
                for packet in output_stream.encode():
                    output_container.mux(packet)
    except gr.Error:
        output_path.unlink(missing_ok=True)
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        output_path.unlink(missing_ok=True)
        raise gr.Error(f"Could not process the video: {exc}") from exc

    return str(output_path)


def _build_upload_panel(
    title: str,
    file_types: list[str],
    elem_id: str,
) -> tuple[gr.File, gr.Button]:
    """Create a styled upload panel shared by both application tabs.

    Parameters
    ----------
    title : str
        Heading displayed above the file picker.
    file_types : list[str]
        Extension filters passed to Gradio's file picker.
    elem_id : str
        Stable HTML element id used for styling and test targeting.

    Returns
    -------
    tuple[gr.File, gr.Button]
        File picker and upload button for wiring to a processing callback.
    """
    with gr.Column(scale=1, min_width=420, elem_id=elem_id, elem_classes=["panel-box"]):
        gr.Markdown(f"<div class='panel-heading'>{title}</div>")
        file_input = gr.File(
            label="Select File",
            file_types=file_types,
            type="filepath",
        )
        upload_button = gr.Button("Upload", variant="primary")
    return file_input, upload_button


def build_app(repository: ModelRepository, segmenter: SemanticSegmenter) -> gr.Blocks:
    """Construct the Gradio interface for model-selectable segmentation.

    Parameters
    ----------
    repository : ModelRepository
        Checkpoint repository supplying dropdown choices and the default value.
    segmenter : SemanticSegmenter
        Shared segmentation service used by both tab callbacks.

    Returns
    -------
    gradio.Blocks
        Fully wired Gradio application ready to launch.
    """
    # Blocks provides enough layout control to reproduce the title/dropdown and
    # side-by-side upload/result panels from the supplied mockup.
    with gr.Blocks(title="Segmented ARD", fill_width=True) as demo:
        gr.Markdown("# Segmented ARD", elem_id="app-title")
        model_dropdown = gr.Dropdown(
            choices=repository.choices,
            value=repository.default_choice,
            label="Model checkpoint",
            show_label=False,
            interactive=True,
            elem_id="model-selector",
        )

        with gr.Tabs(elem_id="workspace"):
            with gr.Tab("Image"):
                with gr.Row(equal_height=True):
                    image_file, image_button = _build_upload_panel(
                        title="Upload",
                        file_types=[".jpg"],
                        elem_id="image-upload-panel",
                    )
                    with gr.Column(scale=1, min_width=420, elem_classes=["panel-box"]):
                        gr.Markdown("<div class='panel-heading'>Result</div>")
                        image_result = gr.Image(
                            label="Segmented Image",
                            type="pil",
                            interactive=False,
                            elem_classes=["result-frame"],
                        )

            with gr.Tab("Video"):
                with gr.Row(equal_height=True):
                    video_file, video_button = _build_upload_panel(
                        title="Upload",
                        file_types=[".mp4", ".mov", ".mkv"],
                        elem_id="video-upload-panel",
                    )
                    with gr.Column(scale=1, min_width=420, elem_classes=["panel-box"]):
                        gr.Markdown("<div class='panel-heading'>Result</div>")
                        video_result = gr.Video(
                            label="Segmented Video",
                            interactive=False,
                            elem_classes=["result-frame"],
                        )

        # Bind the segmenter via partial while leaving the model dropdown as a
        # live input, so each click uses exactly the checkpoint shown in the UI.
        image_button.click(
            fn=partial(segment_image, segmenter=segmenter),
            inputs=[image_file, model_dropdown],
            outputs=image_result,
        )
        video_button.click(
            fn=partial(segment_video, segmenter=segmenter),
            inputs=[video_file, model_dropdown],
            outputs=video_result,
        )

    return demo


def main() -> None:
    """Discover checkpoints and launch the Segmented ARD Gradio application.

    Steps
    -----
    1. Select CUDA when available, otherwise CPU.
    2. Discover all top-level ``./model/*.pt`` checkpoints for the dropdown.
    3. Build the shared checkpoint repository and segmentation service.
    4. Launch the queued Gradio application.

    Returns
    -------
    None
        The function launches Gradio and blocks while the server is running.

    Raises
    ------
    RuntimeError
        If ``./model`` is missing or contains no selectable checkpoints.
    """
    # This matches evaluate_performance.py's device selection while allowing the
    # UI to run on CPU-only systems as a fallback.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repository = ModelRepository(model_dir=MODEL_DIR, device=device)
    segmenter = SemanticSegmenter(repository=repository, device=device)
    demo = build_app(repository=repository, segmenter=segmenter)
    demo.queue().launch(css=APP_CSS)


if __name__ == "__main__":
    main()
