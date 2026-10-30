"""Hosted SAM 3.1 client with a deterministic on-disk cache (FR-PER-03, NFR-DET-01).

The hosted endpoint accepts a **text prompt on a crop** only (D-01); box and point
prompts exist in the open-weight release and are not used here. A response is
cached under ``sha256(pixel bytes + prompt + model)`` as NFR-DET-01 specifies, so
a warm cache makes a benchmark run 100% offline.

The key hashes the **pixels**, never an encoded PNG. CI proved that two machines
encode the same pixels to different (still lossless) deflate streams — 3895 bytes
here, 3925 on the runner — which moved the key without the image changing. Pixel
bytes are bit-identical across the runners and are the content the cache is
actually keyed on.

Three modes, and they are not interchangeable:

``off``    SAM is not in the loop. The classical tracker still runs.
``cache``  cache only. A miss raises :class:`SamCacheMiss`; it never falls back
           to the network. This is the benchmark and judge mode.
``live``   call the API. A transport failure is reported, never swallowed; the
           classical tracker keeps the safety loop ticking while SAM is down.

The live response is streamed **special-token text**: one box and one mask per
match, encoded as base85 and decoded with the official ``meta-sam-parser``
package. The JSON fixture in ``fixtures/sam/`` remains a schema fixture for
injected transports; the cache entry records ``provenance`` so no run can claim
a fixture mask came from the API.

Prompt audit (honest limitation): the configured coolant prompt
(``fluorescent yellow liquid droplet``) is the default and stays the default.
The 2026-10-31 live audit (``artifacts/sam_live_audit.json``) returned **zero
matches** for that prompt and for ``bright green blob`` / ``yellow-green glow``
on two synthetic crops. A zero-match lane is a valid result, not an error, and
earlier notes claiming masks for the alternates were not reproduced. The
default is therefore *not* selected on mask yield — picking a prompt because
it produces a mask would be overfitting to the demo clip. Use
:meth:`SamClient.evaluate_prompts` to trial alternates offline; every
prompt and its result (or error) is recorded, never silently ranked.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from meta_sam_parser import SegmentationMask, decode_mask_to_raster

Provenance = Literal["api", "fixture"]


class SamError(RuntimeError):
    """Base class for every SAM failure."""


class SamPromptError(ValueError):
    """The prompt exceeds the hosted API's 32-token limit."""


class SamConfigError(SamError):
    """A mode was requested without the configuration or credentials it needs."""


class SamCacheMiss(SamError):
    """``cache`` mode was asked for an image that was never cached."""


class SamResponseError(SamError):
    """The response did not contain a decodable mask."""


class SamAPIError(SamError):
    """The hosted endpoint failed or timed out."""


@dataclass(frozen=True)
class SamMask:
    mask: np.ndarray
    bbox_px: tuple[int, int, int, int]
    score: float


@dataclass(frozen=True)
class SamResult:
    masks: tuple[SamMask, ...]
    provenance: Provenance
    cached: bool
    cache_key: str
    prompt: str
    model: str = ""


@dataclass(frozen=True)
class PromptTrial:
    """One prompt-evaluation record: the prompt and what it produced.

    ``result`` is set on success (including a valid zero-match ``()``);
    ``error`` is set when the prompt failed (transport, timeout, malformed
    response). Exactly one of the two is set. Trials are recorded, never
    ranked: a mask is not evidence a prompt is better.
    """

    prompt: str
    model: str
    result: SamResult | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.result is not None

    @property
    def mask_count(self) -> int:
        return len(self.result.masks) if self.result is not None else 0


def count_prompt_tokens(prompt: str) -> int:
    """Conservative client-side token count for the 32-token guard.

    This is not the server tokenizer. It splits words, numbers, and punctuation
    into separate units, so it over-counts rather than under-counts: rejecting a
    prompt the server would have accepted is harmless, the reverse is a 400.
    """

    if not prompt.strip():
        raise SamPromptError("prompt must not be empty")
    return len(re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", prompt))


def validate_prompt(prompt: str, max_tokens: int) -> None:
    tokens = count_prompt_tokens(prompt)
    if tokens > max_tokens:
        raise SamPromptError(f"prompt is {tokens} tokens, limit is {max_tokens}: {prompt!r}")


def cache_key(image_bgr: np.ndarray, prompt: str, model: str) -> str:
    """NFR-DET-01: sha256(pixel bytes + prompt_str + model_name).

    Raw pixels, in C order. Hashing an encoded PNG would make the key depend on
    the compressor's runtime dispatch: the same crop encodes to different bytes
    on different CPUs, so a cache seeded on one machine missed on another while
    the image itself was byte-identical.
    """

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(image_bgr).tobytes())
    digest.update(prompt.encode("utf-8"))
    digest.update(model.encode("utf-8"))
    return digest.hexdigest()


def encode_png(image_bgr: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise SamError("OpenCV could not encode the crop as PNG")
    return buffer.tobytes()


def encode_mask_png(mask: np.ndarray) -> bytes:
    binary = np.where(mask.astype(bool), 255, 0).astype(np.uint8)
    ok, buffer = cv2.imencode(".png", binary, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise SamError("OpenCV could not encode the mask as PNG")
    return buffer.tobytes()


def _decode_base64(text: str) -> bytes:
    payload = text.split(",", 1)[-1] if text.startswith("data:") else text
    try:
        return base64.b64decode(payload, validate=True)
    except (ValueError, TypeError) as exc:
        raise SamResponseError(f"mask is not valid base64: {exc}") from exc


def _mask_from_entry(entry: dict[str, Any], shape: tuple[int, int], threshold: int) -> SamMask:
    encoded = (
        entry.get("mask_png_base64")
        or entry.get("png_base64")
        or entry.get("mask")
        or entry.get("image")
    )
    if not isinstance(encoded, str):
        raise SamResponseError(f"mask entry has no base64 payload: {sorted(entry)}")

    buffer = np.frombuffer(_decode_base64(encoded), dtype=np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise SamResponseError("mask payload is not a decodable image")
    if decoded.ndim != 2:
        raise SamResponseError(f"mask payload is not a 2-D image: ndim={decoded.ndim}")

    if decoded.shape != shape:
        decoded = cv2.resize(decoded, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = validate_crop_mask(decoded >= int(threshold), shape)

    bbox = entry.get("bbox_px")
    if bbox is None:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            bbox = [0, 0, 0, 0]
        else:
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    try:
        if len(bbox) != 4:
            raise SamResponseError(f"bbox_px must have four values, got {bbox!r}")
        x0, y0, x1, y1 = (int(v) for v in bbox)
    except SamResponseError:
        raise
    except (TypeError, ValueError) as exc:
        raise SamResponseError(f"bbox_px is malformed: {bbox!r} ({exc})") from exc
    x0, x1 = sorted((max(0, x0), min(shape[1], x1)))
    y0, y1 = sorted((max(0, y0), min(shape[0], y1)))

    try:
        score = float(entry.get("score", 1.0))
    except (TypeError, ValueError) as exc:
        raise SamResponseError(f"mask score is malformed: {entry.get('score')!r}") from exc
    if not np.isfinite(score):
        raise SamResponseError(f"mask score is not finite: {entry.get('score')!r}")
    return SamMask(mask=mask, bbox_px=(x0, y0, x1, y1), score=max(0.0, min(1.0, score)))


def validate_crop_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Validate a crop-sized mask before it may touch the pipeline.

    Returns a 2-D boolean array of exactly ``shape``. Anything else —
    wrong rank, wrong extent, non-array — is a :class:`SamResponseError`,
    never a silent reshape: a malformed mask must fail loudly at the
    boundary instead of corrupting tracker state downstream.
    """

    if not isinstance(mask, np.ndarray):
        raise SamResponseError(f"mask is not an array: {type(mask).__name__}")
    if mask.ndim != 2:
        raise SamResponseError(f"mask must be 2-D, got ndim={mask.ndim} shape={mask.shape}")
    if mask.shape != shape:
        raise SamResponseError(f"mask shape {mask.shape} does not match crop {shape}")
    try:
        return np.ascontiguousarray(mask).astype(bool)
    except (TypeError, ValueError) as exc:
        raise SamResponseError(f"mask dtype {mask.dtype} is not boolean-convertible") from exc


def _mask_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Find the mask list in either a raw object or a Responses-API envelope."""

    if isinstance(payload.get("masks"), list):
        return [entry for entry in payload["masks"] if isinstance(entry, dict)]

    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "sam_mask":
            return [item]
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if not isinstance(text, str):
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and isinstance(parsed.get("masks"), list):
                return [entry for entry in parsed["masks"] if isinstance(entry, dict)]

    raise SamResponseError(
        "no masks found; expected payload['masks'] or Responses output_text JSON "
        f"with a 'masks' list (keys: {sorted(payload)})"
    )


def parse_response(
    payload: dict[str, Any], image_shape: tuple[int, int], threshold: int = 127
) -> tuple[SamMask, ...]:
    entries = _mask_entries(payload)
    return tuple(_mask_from_entry(entry, image_shape, threshold) for entry in entries)


# One record from the live Responses stream. The grammar is documented at
# https://dev.meta.ai/docs/sam/reading-segmentation: object id, box token, mask
# token. The mask payload is base85 bytes introduced by `~` (lossless) or `!`
# (one_bit); `|` is outside the base85 alphabet used by the API.
_SPECIAL_RECORD = re.compile(
    r"(?P<object_id>\d+)"
    r"<\|box;x1=(?P<x1>\d+);y1=(?P<y1>\d+);x2=(?P<x2>\d+);y2=(?P<y2>\d+);"
    r"w=(?P<frame_w>\d+);h=(?P<frame_h>\d+)(?:;c=(?P<box_c>[0-9]+(?:\.[0-9]+)?))?\|>"
    r"<\|mask;x=0;y=0(?:;c=(?P<mask_c>[0-9]+(?:\.[0-9]+)?))?;"
    r"data=(?P<mask_h>\d+),(?P<mask_w>\d+),(?P<encoded>[~!][^|]*?)\|>"
)


def parse_sam_text(text: str, image_shape: tuple[int, int]) -> tuple[SamMask, ...]:
    """Parse the live special-token lane into masks.

    Zero matches is a valid outcome and is represented by an empty lane. A
    non-empty lane with no parsable record is an error, so a refusal or a schema
    drift cannot be mistaken for "nothing found".
    """

    if not text.strip():
        return ()

    masks: list[SamMask] = []
    for match in _SPECIAL_RECORD.finditer(text):
        height = int(match.group("mask_h"))
        width = int(match.group("mask_w"))
        encoded = match.group("encoded")
        encoding: Literal["lossless", "one_bit"] = (
            "lossless" if encoded.startswith("~") else "one_bit"
        )
        try:
            raster = decode_mask_to_raster(
                SegmentationMask(
                    encoding=encoding,
                    payload=encoded,
                    width=width,
                    height=height,
                )
            )
        except Exception as exc:  # the official decoder raises its own error type
            raise SamResponseError(f"mask payload did not decode: {exc}") from exc

        if len(raster) != width * height:
            raise SamResponseError(
                f"mask raster is {len(raster)} bytes, expected "
                f"{width * height} for {height}x{width}"
            )
        mask = np.frombuffer(raster, dtype=np.uint8).reshape(height, width) > 0
        if mask.shape != image_shape:
            mask = (
                cv2.resize(
                    mask.astype(np.uint8),
                    (image_shape[1], image_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )
        mask = validate_crop_mask(mask, image_shape)

        x0, y0, x1, y1 = (
            int(match.group("x1")),
            int(match.group("y1")),
            int(match.group("x2")),
            int(match.group("y2")),
        )
        x0, x1 = sorted((max(0, x0), min(image_shape[1], x1)))
        y0, y1 = sorted((max(0, y0), min(image_shape[0], y1)))
        score = float(match.group("mask_c") or match.group("box_c") or 1.0)
        masks.append(
            SamMask(
                mask=mask,
                bbox_px=(x0, y0, x1, y1),
                score=max(0.0, min(1.0, score)),
            )
        )

    if not masks:
        raise SamResponseError("SAM output had text but no box/mask record")
    return tuple(masks)


class SamClient:
    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        mode: str | None = None,
        cache_dir: str | Path | None = None,
        api_key: str | None = None,
        transport: Any = None,
    ) -> None:
        self.cfg = cfg
        self.model = str(cfg["model"])
        self.mode = str(mode or cfg.get("mode", "off"))
        if self.mode not in {"off", "cache", "live"}:
            raise SamConfigError(f"sam.mode must be off, cache or live: {self.mode!r}")

        self.cache_dir = Path(cache_dir or cfg.get("cache_dir", "artifacts/sam_cache"))
        self.prompt_max_tokens = int(cfg["prompt_max_tokens"])
        self.timeout_s = float(cfg["timeout_sec"])
        self.threshold = int(cfg.get("mask_threshold", 127))
        self._transport = transport
        self._client: Any = None
        self._api_key = api_key or os.environ.get("META_API_KEY")

        if self.mode == "live" and self._transport is None and not self._api_key:
            raise SamConfigError("sam.mode=live needs META_API_KEY (or an injected transport)")

    # -- cache ---------------------------------------------------------------

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.cache_dir / f"{key}.json", self.cache_dir / f"{key}.png"

    def load_cached(self, key: str, image_shape: tuple[int, int]) -> SamResult | None:
        metadata_path, mask_path = self._paths(key)
        if not metadata_path.is_file():
            # Retry only when the mask file and the JSON agree on the key; a
            # half-written entry is a cache corruption, not a miss.
            if mask_path.is_file():
                raise SamError(f"cache metadata is missing for {mask_path.name}")
            return None

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("key") != key:
            raise SamError(f"cache key mismatch in {metadata_path}")
        provenance = metadata.get("provenance")
        if provenance not in ("api", "fixture"):
            raise SamError(f"cache entry has no provenance: {metadata_path}")

        masks: list[SamMask] = []
        for entry in metadata.get("masks", []):
            name = Path(str(entry.get("mask_file", ""))).name
            mask_file = self.cache_dir / name
            image = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise SamError(f"cache mask is unreadable: {mask_file}")
            binary = validate_crop_mask(image >= self.threshold, image_shape)
            try:
                bbox = tuple(int(v) for v in entry["bbox_px"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SamError(f"cache entry has a malformed bbox_px: {mask_file}") from exc
            if len(bbox) != 4:
                raise SamError(f"cache entry has a malformed bbox_px: {mask_file}")
            try:
                score = float(entry.get("score", 1.0))
            except (TypeError, ValueError) as exc:
                raise SamError(f"cache entry has a malformed score: {mask_file}") from exc
            masks.append(
                SamMask(
                    mask=binary,
                    bbox_px=bbox,
                    score=max(0.0, min(1.0, score)),
                )
            )

        return SamResult(
            masks=tuple(masks),
            provenance=provenance,
            cached=True,
            cache_key=key,
            prompt=str(metadata.get("prompt", "")),
            model=str(metadata.get("model", self.model)),
        )

    def store(
        self,
        key: str,
        prompt: str,
        masks: tuple[SamMask, ...],
        provenance: Provenance,
        image_shape: tuple[int, int],
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        metadata_path, _ = self._paths(key)

        entries: list[dict[str, Any]] = []
        for index, sam_mask in enumerate(masks):
            name = f"{key}.png" if len(masks) == 1 else f"{key}-{index:03d}.png"
            mask_path = self.cache_dir / name
            temporary = mask_path.with_suffix(".png.tmp")
            temporary.write_bytes(encode_mask_png(sam_mask.mask))
            temporary.replace(mask_path)
            entries.append(
                {
                    "bbox_px": list(sam_mask.bbox_px),
                    "score": sam_mask.score,
                    "mask_file": name,
                }
            )

        # The JSON is written last: its presence is what makes the entry valid.
        payload = {
            "schema_version": "1.0",
            "key": key,
            "model": self.model,
            "prompt": prompt,
            "provenance": provenance,
            "image": {"width": image_shape[1], "height": image_shape[0]},
            "masks": entries,
        }
        temporary_json = metadata_path.with_suffix(".json.tmp")
        temporary_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_json.replace(metadata_path)

    # -- api -----------------------------------------------------------------

    def _api(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key,
                base_url=str(self.cfg["base_url"]),
                timeout=self.timeout_s,
                max_retries=int(self.cfg.get("max_retries", 1)),
            )
        return self._client

    def _call_api(self, image_bgr: np.ndarray, image_bytes: bytes, prompt: str) -> str:
        data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        try:
            chunks: list[str] = []
            completed = ""
            with self._api().responses.stream(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    }
                ],
            ) as stream:
                for event in stream:
                    event_type = str(getattr(event, "type", ""))
                    if event_type == "response.output_text.delta":
                        chunks.append(str(getattr(event, "delta", "")))
                    elif "refusal" in event_type:
                        raise SamResponseError("SAM refused the segmentation request")
                    elif event_type == "response.completed":
                        response = getattr(event, "response", None)
                        completed = str(getattr(response, "output_text", "") or "")
            return "".join(chunks) or completed
        except SamError:
            raise
        except Exception as exc:  # openai raises a family of transport errors
            raise SamAPIError(f"{type(exc).__name__}: {exc}") from exc

    # -- public --------------------------------------------------------------

    def segment(self, image_bgr: np.ndarray, prompt: str) -> SamResult:
        if self.mode == "off":
            raise SamConfigError("SAM is disabled; set sam.mode to cache or live")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise SamError(f"expected BGR HxWx3 crop, got {image_bgr.shape}")

        validate_prompt(prompt, self.prompt_max_tokens)
        # The wire still carries a PNG; the key deliberately does not.
        image_bytes = encode_png(image_bgr)
        key = cache_key(image_bgr, prompt, self.model)

        cached = self.load_cached(key, image_bgr.shape[:2])
        if cached is not None:
            return cached
        if self.mode == "cache":
            raise SamCacheMiss(
                f"no cached SAM response for {key}; run scripts/seed_sam_cache.py "
                f"or switch sam.mode to live"
            )

        if self._transport is not None:
            payload = self._transport(image_bytes, prompt)
            masks = parse_response(payload, image_bgr.shape[:2], self.threshold)
        else:
            text = self._call_api(image_bgr, image_bytes, prompt)
            masks = parse_sam_text(text, image_bgr.shape[:2])
        self.store(key, prompt, masks, "api", image_bgr.shape[:2])
        return SamResult(
            masks=masks,
            provenance="api",
            cached=False,
            cache_key=key,
            prompt=prompt,
            model=self.model,
        )

    def evaluate_prompts(
        self, image_bgr: np.ndarray, prompts: list[str]
    ) -> tuple[PromptTrial, ...]:
        """Trial several prompts against one crop, recording every outcome.

        Each prompt is segmented independently; a per-prompt failure is
        recorded in its trial, never raised, so one bad prompt cannot hide
        the rest. Any exception is recorded — including an over-token prompt
        (:class:`SamPromptError` is a ``ValueError``, not a ``SamError``)
        and a raw transport error — because a trial that raised is still a
        trial outcome, not a reason to stop the audit. The caller's
        configured default is unchanged — the trials are an audit, not a
        selection: a mask is not evidence a prompt is better, and an empty
        (zero-match) lane is a valid result, not an error to optimise away.
        """

        trials: list[PromptTrial] = []
        for prompt in prompts:
            try:
                result = self.segment(image_bgr, prompt)
            except Exception as exc:
                trials.append(PromptTrial(prompt=prompt, model=self.model, error=repr(exc)))
                continue
            trials.append(PromptTrial(prompt=prompt, model=self.model, result=result))
        return tuple(trials)
