#!/bin/bash
# Helper script to start Jupyter Notebook with the correct uv environment

echo "Starting Jupyter Notebook with NLP Project environment..."
echo "The notebook will open in your browser automatically."
echo "Press Ctrl+C to stop the server when you're done."
echo ""

uv run jupyter notebook tos_classification.ipynb
