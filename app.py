from __future__ import annotations

import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Final, cast

import albumentations as A
import av  # type: ignore[import-untyped]
import numpy as np
import streamlit as st
import torch
from albumentations.pytorch import ToTensorV2
from beartype import beartype
from jaxtyping import Float, UInt8, jaxtyped
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile
from torch import Tensor, nn

IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset({".jpg"})
VIDEO_SUFFIXES: Final[frozenset[str]] = frozenset({".mp4", ".mov", ".mkv"})
MODEL_SUFFIX: Final[str] = ".pt"
MODEL_DIR: Final[Path] = Path("./model")
MASK_ALPHA: Final[float] = 0.5
DEFAULT_VIDEO_RATE: Final[Fraction] = Fraction(30, 1)
IMAGE_HEIGHT: Final[int] = 360
IMAGE_WIDTH: Final[int] = 640
# Streamlit defaults uploads to 200 MB, while the previous Gradio launch had no
# configured cap. A 1-TiB practical ceiling avoids introducing a normal-use limit.
STREAMLIT_MAX_UPLOAD_SIZE_MB: Final[int] = 1024 * 1024

# Keep the palette identical to evaluate_performance.py so predictions shown in
# the Streamlit app use the same semantic colours as the project's evaluation
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
<style>
.block-container {
    max-width: 1500px;
    padding: 24px 3vw 48px 3vw;
}
#app-title {
    text-align: center;
    margin: 12px 0 10px 0;
}
.model-selector-label {
    text-align: center;
    color: rgba(49, 51, 63, 0.65);
    margin-bottom: 4px;
}
.panel-heading {
    text-align: center;
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 12px;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 24px;
}
div[data-testid="stImage"], div[data-testid="stVideo"] {
    min-height: 300px;
}
.checkpoint-used {
    margin-top: 10px;
    text-align: center;
    font-size: 0.95rem;
    color: rgba(49, 51, 63, 0.65);
}
</style>
"""


class ApplicationError(RuntimeError):
    """Represent a user-facing application error that Streamlit can display safely.

    The exception keeps validation, model-loading, inference, and encoding failures
    separate from Streamlit rendering. UI functions catch it and show only the
    actionable message instead of exposing an internal traceback to the user.

    Parameters
    ----------
    message : str
        Human-readable explanation of the failure.
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

    The repository exposes all top-level ``.pt`` files to the Streamlit dropdown.
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
        ApplicationError
            If the selection is invalid, the checkpoint cannot be read, or the
            deserialized object is not an ``nn.Module``.
        """
        path = self._choice_to_path.get(checkpoint_choice)
        if path is None:
            raise ApplicationError("Select a valid model checkpoint from the dropdown.")

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
            raise ApplicationError(f"Could not load checkpoint {path.name}: {exc}") from exc

        if not isinstance(checkpoint, nn.Module):
            raise ApplicationError(
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
        ApplicationError
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
        except ApplicationError:
            raise
        except (IndexError, RuntimeError, TypeError, ValueError) as exc:
            raise ApplicationError(f"Segmentation inference failed: {exc}") from exc

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
        Filesystem path returned by the Streamlit file component.
    allowed_suffixes : frozenset[str]
        Lowercase file suffixes accepted for the active tab.

    Returns
    -------
    pathlib.Path
        Validated path object for downstream processing.

    Raises
    ------
    ApplicationError
        If the file extension is not allowed or the uploaded path is missing.
    """
    path = Path(file_path)

    # Browser picker filters improve UX but are not a validation boundary, so
    # enforce the extension again on the server before decoding user content.
    if path.suffix.lower() not in allowed_suffixes:
        allowed = ", ".join(sorted(allowed_suffixes))
        raise ApplicationError(f"Unsupported file type. Allowed extensions: {allowed}")
    if not path.is_file():
        raise ApplicationError("The uploaded file is no longer available on the server.")
    return path


def segment_image(
    file_path: str,
    checkpoint_choice: str,
    segmenter: SemanticSegmenter,
) -> tuple[Image.Image, str]:
    """Validate and segment an uploaded JPG with the selected checkpoint.

    Parameters
    ----------
    file_path : str
        Path to the uploaded JPG supplied by Streamlit.
    checkpoint_choice : str
        Dropdown value identifying the model used for inference.
    segmenter : SemanticSegmenter
        Reusable segmentation service shared by the application.

    Returns
    -------
    tuple[PIL.Image.Image, str]
        Original image with its semantic segmentation mask blended on top and a
        short label identifying the checkpoint used for this inference.

    Raises
    ------
    ApplicationError
        If the extension is invalid, Pillow cannot decode the image, or model
        inference fails.
    """
    path = _validate_suffix(file_path, IMAGE_SUFFIXES)
    try:
        # Open within the callback so uploaded-file handles are released as soon
        # as the request completes.
        with Image.open(path) as image:
            segmented_image = segmenter.segment(image, checkpoint_choice)
            return segmented_image, _checkpoint_used_text(checkpoint_choice)
    except ApplicationError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise ApplicationError(f"Could not process the JPG image: {exc}") from exc


def _apply_video_display_rotation(frame: av.VideoFrame) -> Image.Image:
    """Convert a decoded frame to the orientation intended for display.

    MOV/MP4 containers may keep camera pixels in their coded orientation and
    store a display-matrix rotation separately. PyAV exposes that rotation on
    ``VideoFrame.rotation``, but ``VideoFrame.to_image()`` intentionally returns
    the unrotated coded pixels. This helper converts the frame to Pillow and then
    applies the metadata rotation before segmentation and re-encoding.

    Steps
    -----
    1. Convert the decoded PyAV frame to an RGB Pillow image.
    2. Normalize the counter-clockwise display rotation into ``[0, 360)``.
    3. Use exact transpose operations for right-angle rotations so no pixels are
       interpolated before semantic-segmentation inference.
    4. Fall back to Pillow rotation for uncommon non-right-angle metadata.

    Parameters
    ----------
    frame : av.VideoFrame
        Decoded video frame whose display-matrix metadata should be respected.

    Returns
    -------
    PIL.Image.Image
        RGB frame physically rotated into its intended display orientation.
    """
    image = frame.to_image()
    rotation = frame.rotation % 360

    # Display-matrix values generated by cameras are normally quarter turns.
    # PyAV 14/15 exposes the FFmpeg-derived angle as an integer, so values very
    # close to a quarter turn can become 89/179/269 after conversion. Snap only
    # those near-quarter-turn values instead of changing arbitrary rotations.
    nearest_quarter_turn = int(round(rotation / 90.0) * 90) % 360
    circular_distance = min(
        (rotation - nearest_quarter_turn) % 360,
        (nearest_quarter_turn - rotation) % 360,
    )
    if circular_distance <= 1:
        rotation = nearest_quarter_turn

    if rotation == 0:
        return image
    if rotation == 90:
        return image.transpose(Image.Transpose.ROTATE_90)
    if rotation == 180:
        return image.transpose(Image.Transpose.ROTATE_180)
    if rotation == 270:
        return image.transpose(Image.Transpose.ROTATE_270)

    # Arbitrary display rotations are unusual for phone/camera MOV files, but
    # honoring them is safer than silently discarding valid container metadata.
    return image.rotate(
        rotation,
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )


def segment_video(
    file_path: str,
    checkpoint_choice: str,
    segmenter: SemanticSegmenter,
) -> tuple[str, str]:
    """Segment every frame of an uploaded video with one selected checkpoint.

    The video is decoded frame-by-frame with PyAV. Container display rotation is
    applied before every frame goes through the same segmentation path as a still
    image, then the overlays are encoded to H.264 MP4 for browser playback. Audio
    is intentionally omitted because the requested result is the segmented visual
    stream.

    Parameters
    ----------
    file_path : str
        Path to an uploaded MP4, MOV, or MKV video supplied by Streamlit.
    checkpoint_choice : str
        Dropdown value identifying the model used for every video frame.
    segmenter : SemanticSegmenter
        Reusable segmentation service shared by the application.

    Returns
    -------
    tuple[str, str]
        Path to the generated H.264 MP4 displayed by Streamlit and a short label
        identifying the checkpoint used for every frame.

    Raises
    ------
    ApplicationError
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
                raise ApplicationError("The uploaded file does not contain a video stream.")

            input_stream = input_container.streams.video[0]
            frame_rate = input_stream.average_rate or DEFAULT_VIDEO_RATE

            with av.open(str(output_path), mode="w") as output_container:
                # Create the encoder lazily because a MOV display matrix can rotate
                # portrait-coded pixels into a landscape display frame (or vice
                # versa). The first oriented result therefore defines the true
                # output dimensions instead of the raw codec dimensions.
                output_stream = None

                # Resolve/load the checkpoint on the first frame, then the model
                # repository cache keeps every subsequent frame on the same model.
                for frame in input_container.decode(input_stream):
                    oriented_frame = _apply_video_display_rotation(frame)
                    segmented = segmenter.segment(oriented_frame, checkpoint_choice)

                    if output_stream is None:
                        # H.264/yuv420p is broadly supported by web browsers and
                        # Streamlit. Because pixels have already been physically
                        # rotated, no display-rotation metadata is copied to the
                        # output, preventing a second rotation during playback.
                        output_stream = output_container.add_stream(
                            "libx264",
                            rate=frame_rate,
                        )
                        output_stream.width = segmented.width
                        output_stream.height = segmented.height
                        output_stream.pix_fmt = "yuv420p"
                    elif segmented.size != (output_stream.width, output_stream.height):
                        raise ApplicationError(
                            "Video display orientation changed dimensions between frames."
                        )

                    encoded_frame = av.VideoFrame.from_image(segmented)
                    for packet in output_stream.encode(encoded_frame):
                        output_container.mux(packet)

                if output_stream is None:
                    raise ApplicationError("The uploaded video does not contain decodable frames.")

                # Flush delayed encoder packets so the final MP4 is complete and
                # seekable before Streamlit serves it to the browser.
                for packet in output_stream.encode():
                    output_container.mux(packet)
    except ApplicationError:
        output_path.unlink(missing_ok=True)
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        output_path.unlink(missing_ok=True)
        raise ApplicationError(f"Could not process the video: {exc}") from exc

    return str(output_path), _checkpoint_used_text(checkpoint_choice)


def _checkpoint_used_text(checkpoint_choice: str) -> str:
    """Format the checkpoint label displayed beneath an inference result.

    Parameters
    ----------
    checkpoint_choice : str
        Dropdown value identifying the checkpoint used for inference.

    Returns
    -------
    str
        Markdown text showing the exact checkpoint path used for the result.
    """
    # Render the path as inline code so it remains readable even when checkpoint
    # names contain underscores or other Markdown-significant characters.
    return f"Model checkpoint used: `{checkpoint_choice}`"


def _materialize_uploaded_file(
    uploaded_file: UploadedFile,
    allowed_suffixes: frozenset[str],
) -> Path:
    """Write one Streamlit upload to a validated temporary filesystem path.

    Streamlit keeps uploaded content in memory while the existing image/video
    processing functions intentionally operate on paths. This adapter validates
    the client filename first, writes the bytes to a unique temporary file with
    the same suffix, and returns that path for the unchanged processing pipeline.

    Parameters
    ----------
    uploaded_file : streamlit.runtime.uploaded_file_manager.UploadedFile
        Uploaded file supplied by Streamlit's file uploader.
    allowed_suffixes : frozenset[str]
        Lowercase file suffixes accepted for the active application tab.

    Returns
    -------
    pathlib.Path
        Temporary path containing the uploaded bytes.

    Raises
    ------
    ApplicationError
        If the filename suffix is unsupported or the temporary file cannot be
        written.
    """
    suffix = Path(uploaded_file.name).suffix.lower()

    # Streamlit's picker filter is a convenience rather than a security boundary,
    # so reject a mismatched filename before any bytes are written to disk.
    if suffix not in allowed_suffixes:
        allowed = ", ".join(sorted(allowed_suffixes))
        raise ApplicationError(f"Unsupported file type. Allowed extensions: {allowed}")

    temporary_path: Path | None = None
    try:
        # Preserve the suffix because the downstream validation intentionally
        # checks the materialized path again before decoding user-controlled data.
        with tempfile.NamedTemporaryFile(
            prefix="uploaded_",
            suffix=suffix,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(uploaded_file.getbuffer())
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ApplicationError(f"Could not store the uploaded file: {exc}") from exc

    if temporary_path is None:
        # A successful NamedTemporaryFile context always assigns this path; this
        # defensive branch also keeps the return type explicit for static checking.
        raise ApplicationError("Could not create a temporary upload file.")
    return temporary_path


@st.cache_resource(show_spinner=False)
def get_services() -> tuple[ModelRepository, SemanticSegmenter]:
    """Create and cache the model repository and segmentation service.

    Streamlit reruns the script whenever a widget changes. Caching these services
    preserves the original lazy checkpoint cache across reruns instead of
    repeatedly rebuilding the repository and reloading already-used models.

    Returns
    -------
    tuple[ModelRepository, SemanticSegmenter]
        Shared checkpoint repository and segmentation service for this process.

    Raises
    ------
    RuntimeError
        If ``./model`` is missing or contains no selectable checkpoints.
    """
    # Match the evaluation pipeline's CUDA preference while retaining the
    # original CPU fallback for environments without an available GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repository = ModelRepository(model_dir=MODEL_DIR, device=device)
    segmenter = SemanticSegmenter(repository=repository, device=device)
    return repository, segmenter


def _render_checkpoint_used(checkpoint_text: str) -> None:
    """Render the checkpoint label beneath a completed inference result.

    Parameters
    ----------
    checkpoint_text : str
        Markdown label returned by the image or video processing function.

    Returns
    -------
    None
        The function writes the label directly to the active Streamlit container.
    """
    # Caption supports Markdown, so the checkpoint path keeps its inline-code
    # formatting without relying on fragile cross-call HTML wrappers.
    st.caption(checkpoint_text)


def _render_image_tab(
    checkpoint_choice: str,
    segmenter: SemanticSegmenter,
) -> None:
    """Render and process the still-image segmentation workflow.

    The tab keeps the original explicit Upload action: the selected file is only
    segmented when the user clicks the button. Successful results and their model
    labels are stored in session state so unrelated Streamlit reruns do not erase
    the last completed inference.

    Parameters
    ----------
    checkpoint_choice : str
        Dropdown value identifying the model checkpoint for the next inference.
    segmenter : SemanticSegmenter
        Shared segmentation service used to process uploaded JPG files.

    Returns
    -------
    None
        The function renders controls and results directly into the current tab.
    """
    upload_column, result_column = st.columns(2, gap="large")

    with upload_column:
        with st.container(border=True):
            st.markdown("<div class='panel-heading'>Upload</div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Select File",
                type=["jpg"],
                key="image-file-uploader",
                max_upload_size=STREAMLIT_MAX_UPLOAD_SIZE_MB,
            )
            process_clicked = st.button(
                "Upload",
                key="image-upload-button",
                type="primary",
                width="stretch",
                disabled=uploaded_file is None,
            )

            if process_clicked and uploaded_file is not None:
                temporary_path: Path | None = None
                try:
                    # Materialize only for the duration of processing so uploaded
                    # data does not accumulate on the server across reruns.
                    temporary_path = _materialize_uploaded_file(
                        uploaded_file,
                        IMAGE_SUFFIXES,
                    )
                    with st.spinner("Segmenting image..."):
                        image_result, checkpoint_text = segment_image(
                            str(temporary_path),
                            checkpoint_choice,
                            segmenter,
                        )
                    st.session_state["image-result"] = image_result
                    st.session_state["image-checkpoint"] = checkpoint_text
                except ApplicationError as exc:
                    st.error(str(exc))
                finally:
                    if temporary_path is not None:
                        temporary_path.unlink(missing_ok=True)

    with result_column:
        with st.container(border=True):
            st.markdown("<div class='panel-heading'>Result</div>", unsafe_allow_html=True)
            image_result = st.session_state.get("image-result")
            if isinstance(image_result, Image.Image):
                st.image(image_result, caption="Segmented Image", width="stretch")
                checkpoint_text = st.session_state.get("image-checkpoint")
                if isinstance(checkpoint_text, str):
                    _render_checkpoint_used(checkpoint_text)
            else:
                st.info("Upload a JPG image and click Upload to view the result.")


def _render_video_tab(
    checkpoint_choice: str,
    segmenter: SemanticSegmenter,
) -> None:
    """Render and process the video segmentation workflow.

    The tab preserves the original MP4/MOV/MKV inputs and H.264 MP4 output. The
    generated video is read into session state and its temporary output file is
    removed immediately, allowing the last successful result to survive normal
    Streamlit reruns without leaking output files.

    Parameters
    ----------
    checkpoint_choice : str
        Dropdown value identifying the model checkpoint for every video frame.
    segmenter : SemanticSegmenter
        Shared segmentation service used to process uploaded videos.

    Returns
    -------
    None
        The function renders controls and results directly into the current tab.
    """
    upload_column, result_column = st.columns(2, gap="large")

    with upload_column:
        with st.container(border=True):
            st.markdown("<div class='panel-heading'>Upload</div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Select File",
                type=["mp4", "mov", "mkv"],
                key="video-file-uploader",
                max_upload_size=STREAMLIT_MAX_UPLOAD_SIZE_MB,
            )
            process_clicked = st.button(
                "Upload",
                key="video-upload-button",
                type="primary",
                width="stretch",
                disabled=uploaded_file is None,
            )

            if process_clicked and uploaded_file is not None:
                temporary_path: Path | None = None
                output_path: Path | None = None
                try:
                    # PyAV requires a real path for the existing decoding pipeline,
                    # so bridge Streamlit's in-memory upload only for this request.
                    temporary_path = _materialize_uploaded_file(
                        uploaded_file,
                        VIDEO_SUFFIXES,
                    )
                    with st.spinner("Segmenting video..."):
                        output_path_text, checkpoint_text = segment_video(
                            str(temporary_path),
                            checkpoint_choice,
                            segmenter,
                        )
                        output_path = Path(output_path_text)
                        video_bytes = output_path.read_bytes()
                    st.session_state["video-result"] = video_bytes
                    st.session_state["video-checkpoint"] = checkpoint_text
                except ApplicationError as exc:
                    st.error(str(exc))
                except OSError as exc:
                    st.error(f"Could not read the segmented video: {exc}")
                finally:
                    if temporary_path is not None:
                        temporary_path.unlink(missing_ok=True)
                    if output_path is not None:
                        output_path.unlink(missing_ok=True)

    with result_column:
        with st.container(border=True):
            st.markdown("<div class='panel-heading'>Result</div>", unsafe_allow_html=True)
            video_result = st.session_state.get("video-result")
            if isinstance(video_result, bytes):
                st.video(video_result, format="video/mp4")
                checkpoint_text = st.session_state.get("video-checkpoint")
                if isinstance(checkpoint_text, str):
                    _render_checkpoint_used(checkpoint_text)
            else:
                st.info("Upload an MP4, MOV, or MKV video and click Upload to view the result.")


def build_app(repository: ModelRepository, segmenter: SemanticSegmenter) -> None:
    """Construct the Streamlit interface for model-selectable segmentation.

    Parameters
    ----------
    repository : ModelRepository
        Checkpoint repository supplying dropdown choices and the default value.
    segmenter : SemanticSegmenter
        Shared segmentation service used by the image and video tabs.

    Returns
    -------
    None
        The function renders the complete Streamlit page for the current rerun.
    """
    # Inject only stable page-level styling and keep all behavior in native
    # Streamlit widgets so the migration does not depend on browser-side scripts.
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown("<h1 id='app-title'>Segmented ARD</h1>", unsafe_allow_html=True)

    # Center the model selector similarly to the previous UI while leaving its
    # value live so each Upload click uses exactly the checkpoint shown here.
    _, selector_column, _ = st.columns([1, 1.4, 1])
    with selector_column:
        st.markdown(
            "<div class='model-selector-label'>Model checkpoint</div>",
            unsafe_allow_html=True,
        )
        checkpoint_choice = st.selectbox(
            "Model checkpoint",
            options=repository.choices,
            index=repository.choices.index(repository.default_choice),
            label_visibility="collapsed",
            key="model-checkpoint-selector",
        )

    image_tab, video_tab = st.tabs(["Image", "Video"])
    with image_tab:
        _render_image_tab(checkpoint_choice, segmenter)
    with video_tab:
        _render_video_tab(checkpoint_choice, segmenter)


def main() -> None:
    """Configure and render the Segmented ARD Streamlit application.

    Steps
    -----
    1. Configure the page for a wide two-panel layout.
    2. Create or reuse the cached checkpoint repository and segmentation service.
    3. Render the model selector plus image and video processing tabs.

    Returns
    -------
    None
        The function renders one Streamlit application rerun.
    """
    # Page configuration must be the first Streamlit command executed on each run.
    st.set_page_config(page_title="Segmented ARD", layout="wide")

    try:
        repository, segmenter = get_services()
    except RuntimeError as exc:
        # A missing model directory makes inference impossible, so show a concise
        # startup error instead of allowing Streamlit to expose a stack trace.
        st.error(str(exc))
        return

    build_app(repository=repository, segmenter=segmenter)


if __name__ == "__main__":
    main()
