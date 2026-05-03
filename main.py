#!/usr/bin/env python3
import os
import sys

# -------------------------------------------------------------------
# 1. Define the directory structure and files
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DIRECTORIES = [
    "src",
    "data_sample",
    "results",
    "figures"
]

FILES = [
    "README.md",
    "LICENSE"
]


# -------------------------------------------------------------------
# 2. Helper functions
# -------------------------------------------------------------------
def create_directory(path):
    """Create a directory if it does not exist."""
    try:
        os.makedirs(path, exist_ok=True)
        print(f"✅ Created directory: {path}")
    except Exception as e:
        print(f"❌ Failed to create directory {path}: {e}")
        sys.exit(1)


def create_file(path, content=""):
    """Create a file with optional content."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Created file: {path}")
    except Exception as e:
        print(f"❌ Failed to create file {path}: {e}")
        sys.exit(1)


# -------------------------------------------------------------------
# 3. Main creation routine
# -------------------------------------------------------------------
def main():
    print("🚀 Setting up AutoGradeNet project structure...\n")

    # Create directories
    for d in DIRECTORIES:
        full_path = os.path.join(PROJECT_ROOT, d)
        create_directory(full_path)

    # Create README.md (minimal template)
    readme_content = """# AutoGradeNet: A Novel Fully Autonomous Machine Learning Framework for Human‑Free Multi‑Dimensional Student Assessment and Academic Performance Prediction

## Project Structure
- `/src` – Source code for data generation, model training, evaluation, and fairness analysis.
- `/data_sample` – Small sample datasets for quick testing (e.g., 100 students).
- `/results` – Output logs, trained models, evaluation metrics, and performance tables.
- `/figures` – Generated plots (confusion matrices, learning curves, feature importance, etc.).

## Getting Started
1. Run `generate_synthetic_data.py` (if you have the full data generator) to create raw data.
2. Place any real or additional data in `/data_sample`.
3. Execute training and evaluation scripts from `/src`.

## Dependencies
- Python 3.8+
- numpy, pandas, scikit-learn, matplotlib, seaborn

## License
This project is licensed under the terms of the MIT License (see LICENSE file).
"""
    create_file(os.path.join(PROJECT_ROOT, "README.md"), readme_content)

    # Create LICENSE file (MIT License)
    license_content = """MIT License

Copyright (c) 2026 AutoGradeNet Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    create_file(os.path.join(PROJECT_ROOT, "LICENSE"), license_content)

    # Optionally, create a placeholder .gitkeep inside empty directories
    # to ensure they are tracked by version control.
    for d in DIRECTORIES:
        gitkeep_path = os.path.join(PROJECT_ROOT, d, ".gitkeep")
        if not os.path.exists(gitkeep_path):
            create_file(gitkeep_path, "# This file ensures the directory is tracked by Git.\n")

    print("\n✨ AutoGradeNet project structure created successfully!")


if __name__ == "__main__":
    main()