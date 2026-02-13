"""
Patch whisperx to support pyannote-audio >= 4.0 (based on PR #1344).
Run after `pip install whisperx` to apply compatibility fixes.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path


def _patch_file(path: Path, replacements: list[tuple[str, str]]) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in replacements:
        text = text.replace(old, new)
    if text != original:
        path.write_text(text, encoding="utf-8")
        print(f"  patched: {path}")
        return True
    print(f"  skipped (already patched or no match): {path}")
    return False


def main() -> None:
    import whisperx
    pkg_dir = Path(whisperx.__file__).parent

    print("Patching whisperx for pyannote-audio 4.x ...")

    # 1) __main__.py: default diarize model
    _patch_file(
        pkg_dir / "__main__.py",
        [
            (
                'default="pyannote/speaker-diarization-3.1"',
                'default="pyannote/speaker-diarization-community-1"',
            ),
        ],
    )

    # 2) diarize.py: token= instead of use_auth_token=, new output API
    diarize = pkg_dir / "diarize.py"
    _patch_file(
        diarize,
        [
            # from_pretrained token kwarg
            (
                "Pipeline.from_pretrained(model_config, use_auth_token=use_auth_token)",
                "Pipeline.from_pretrained(\n            model_config, token=use_auth_token\n        )",
            ),
            # new output API — replace the old branching logic
            (
                """if return_embeddings:
            diarization, embeddings = self.model(
                audio_data,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_embeddings=True,
            )
        else:
            diarization = self.model(
                audio_data,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            embeddings = None""",
                """output = self.model(
            audio_data,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        diarization = output.speaker_diarization
        embeddings = output.speaker_embeddings""",
            ),
        ],
    )

    # 3) vads/pyannote.py: token= instead of use_auth_token=
    vad = pkg_dir / "vads" / "pyannote.py"
    _patch_file(
        vad,
        [
            (
                "Model.from_pretrained(model_fp, use_auth_token=use_auth_token)",
                "Model.from_pretrained(model_fp, token=use_auth_token)",
            ),
            (
                "super().__init__(segmentation=segmentation, fscore=fscore, use_auth_token=use_auth_token,",
                "super().__init__(segmentation=segmentation, fscore=fscore, token=use_auth_token,",
            ),
        ],
    )

    print("Done.")


if __name__ == "__main__":
    main()
