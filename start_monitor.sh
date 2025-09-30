#!/bin/bash

# Activate virtual environment and start monitoring
source .venv/bin/activate
python -m src.monitor --update_interval 3
