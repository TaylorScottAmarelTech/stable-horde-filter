from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="stable-horde-filter",
    version="0.1.0",
    author="Taylor Scott Amarel Tech",
    author_email="your.email@example.com",
    description="Image validation filter for Stable Horde",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TaylorScottAmarelTech/stable-horde-filter",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "Pillow>=8.0.0",
        "scipy>=1.5.0",
    ],
    extras_require={
        "ocr": ["pytesseract>=0.3.8"],
        "full": ["opencv-python>=4.5.0", "pytesseract>=0.3.8"],
    },
)