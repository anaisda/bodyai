"""
Setup script for AI Body Measurement Application
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = requirements_path.read_text(encoding="utf-8").strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]
else:
    requirements = [
        "tensorflow>=2.15.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",
        "mediapipe>=0.10.0",
        "opencv-python>=4.8.0",
        "scikit-image>=0.21.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "customtkinter>=5.2.0",
        "matplotlib>=3.7.0",
        "plotly>=5.15.0",
        "albumentations>=1.3.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
    ]

setup(
    name="ai-body-measurement",
    version="1.0.0",
    author="AI Body Measurement Team",
    author_email="contact@aibodymeasurement.com",
    description="Advanced AI-powered body measurement application using computer vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ai-body-measurement/ai-body-measurement",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": [
            "tensorflow-gpu>=2.15.0",
            "cupy-cuda11x>=12.0.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "all": [
            "tensorflow-gpu>=2.15.0",
            "cupy-cuda11x>=12.0.0",
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-body-measurement=main:main",
            "body-measure=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["config/*.yaml", "config/*.json"],
    },
    project_urls={
        "Bug Reports": "https://github.com/ai-body-measurement/ai-body-measurement/issues",
        "Source": "https://github.com/ai-body-measurement/ai-body-measurement",
        "Documentation": "https://ai-body-measurement.readthedocs.io/",
    },
)