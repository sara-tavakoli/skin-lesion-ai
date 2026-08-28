"""Research-grade dermoscopic skin-lesion classification.

Public API is intentionally small; import submodules explicitly, e.g.::

    from skinlesion.data.datamodule import HAM10000DataModule
    from skinlesion.models.classifier import LesionClassifier
"""

from __future__ import annotations

__version__ = "0.1.0"

# Canonical HAM10000 label space (alphabetical by short code) and human-readable
# names.  Melanoma (``mel``) is the clinically critical positive class.
CLASSES: tuple[str, ...] = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")

CLASS_NAMES: dict[str, str] = {
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}

# Lesions that are malignant or pre-malignant -- used for the binary
# "refer to dermatologist" operating-point analysis.
MALIGNANT: frozenset[str] = frozenset({"akiec", "bcc", "mel"})

__all__ = ["CLASSES", "CLASS_NAMES", "MALIGNANT", "__version__"]
