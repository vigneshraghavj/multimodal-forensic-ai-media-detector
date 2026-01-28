Multimodal Forensic AI Media Detector
Overview

This repository contains the implementation of a Multimodal Forensic Framework designed to assess the authenticity of audio and video media that may be generated or manipulated using artificial intelligence techniques such as deepfake synthesis and AI voice cloning.

The tool is developed as a forensic decision-support system, integrating independent audio analysis and video analysis pipelines and combining their outputs through confidence-based multimodal fusion.
Rather than providing absolute judgments, the system emphasizes probabilistic interpretation, aligning with real-world digital forensic practices.

What This Tool Does

The framework performs the following operations:

🔊 Audio Analysis

Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from speech signals

Classifies audio as Human or AI-Generated

Produces a confidence score indicating reliability of the decision

Trained using the ASVspoof 2019 benchmark dataset

🎥 Video Analysis

Detects faces using the YuNet deep learning face detector

Extracts Local Binary Pattern (LBP) texture features from facial regions

Classifies videos as Real or Deepfake

Aggregates predictions across multiple frames using majority voting

Trained using the Celeb-DF-v2 dataset

🔗 Multimodal Fusion

Combines audio and video confidence scores using weighted averaging

Produces a final forensic confidence score

Enables cross-verification when one modality is uncertain

📑 Forensic Reporting

Generates forensic-ready outputs including:

Hash values (MD5, SHA-256)

Confidence graphs

Structured PDF forensic reports

Stores case history in an SQLite database for traceability

Datasets Used
🔊 Audio Datasets

Training Dataset

ASVspoof 2019

Benchmark dataset for spoofed speech detection

Used for training the audio classification model

Testing Dataset

Speech Dataset of Human and AI-Generated Voices

Azis, H.; Rismayanti, N.; Abdullah, M.; Ismail, S.

Mendeley Data, Version 2, 2025

DOI: https://doi.org/10.17632/5czyx2vppv.2

🎥 Video Datasets

Training Dataset

Celeb-DF-v2

Large-scale deepfake video dataset

Used for training the video classification model

Testing Dataset

SDFVD: Small-scale Deepfake Forgery Video Dataset

Kaman, S.; Makandar, A.

Mendeley Data, Version 1, 2024

DOI: https://doi.org/10.17632/bcmkfgct2s.1

Dataset Usage Notice

Due to licensing restrictions, training datasets are not redistributed in this repository.
Users must obtain datasets directly from their original sources.
Instructions for dataset preprocessing and feature extraction are provided within the codebase.

Intended Use & Disclaimer

⚠️ Disclaimer

This tool is intended solely for academic research and forensic decision support.
It does not provide absolute determination of media authenticity and must not be used as the sole basis for legal, judicial, or investigative conclusions.

Final interpretation should always be performed by a qualified forensic expert in conjunction with contextual and corroborative evidence.

Reproducibility

All experiments reported in the associated IEEE conference paper were conducted using this codebase.

Model training, testing, and evaluation pipelines are included.

Confidence-driven outputs are intentionally preserved to reflect realistic forensic uncertainty.

License

This project is released under the MIT License.
See the LICENSE file for full terms.

Citation

If you use this tool or codebase in academic work, please cite the associated research paper:

Vignesh Raghav J, Don Caeiro,
A Multimodal Forensic Framework for Assessing Authenticity of AI-Generated Audio and Video Content,
IEEE Conference Proceedings, 2025.