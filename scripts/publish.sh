#!/bin/bash
# PyPI publish script
# Usage: ./scripts/publish.sh [testpypi|pypi]

set -e

REPOSITORY=${1:-pypi}

echo "Cleaning previous build artifacts..."
rm -rf dist/ build/ *.egg-info

echo "Building package..."
python -m build

echo "Build complete!"

if [ "$REPOSITORY" = "testpypi" ]; then
    echo "Uploading to TestPyPI..."
    twine upload --repository testpypi dist/*
    echo "TestPyPI publish complete!"
    echo "Test installation: pip install --index-url https://test.pypi.org/simple/ pipecat-ai-typecast"
else
    echo "Uploading to PyPI..."
    twine upload dist/*
    echo "PyPI publish complete!"
fi

