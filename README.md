# Crowd Monitoring System

## Description

This project detects and monitors crowd density using computer vision. It provides real-time alerts via sound, email, and SMS when crowd levels exceed thresholds.

## Features

* Real-time people detection
* Crowd counting
* Zone monitoring
* Heatmap visualization
* Email & SMS alerts
* Data logging (CSV)
* Live graph analytics

## Requirements

* Python 3.10
* OpenCV
* NumPy
* Matplotlib
* Twillio
* python-dotenv

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Set environment variables in `.env` file

3. Run:
   main.py

## Output

* Output video file
* detect people and people counting 
* alaram trigger when reach limite
* Heatmap display
* Real-time graph

## Limitations

Uses basic motion detection
Mask detection is simulated

