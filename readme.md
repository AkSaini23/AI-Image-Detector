# Multi-Scale GLDN Feature Extractor

Extracts multi-scale GLDN features from an image using Sobel gradients, directional encoding, and block histograms.

## Run

Keep all files and `Feature_extractor.jpg` in the same folder, then:

```bash
python feature_extractor.py
```

## Files

- `feature_extractor.py` — main script
- `sobel_gradient_operator.py` — gradients (Sec 3.1)
- `gldn_encoder.py` — GLDN encoding (Sec 3.2)
- `multi_gldn_histogram.py` — block histograms (Sec 3.3, Eq 17)

## Reference

Based on [paper](https://dl.acm.org/doi/10.1016/j.jvcir.2023.103876) : Detection of GAN generated image using color gradient representation -- see sections 3.1–3.3.



