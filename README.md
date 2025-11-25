# polygence-website

## Overview

This project detects intonation errors in violin recordings. It analyzes input audio (from a recorded file) and provides feedback on how sharp or flat the played pitch is compared to the target frequency, as well as tips on how the user can practice.

## Features
* Pitch detection using CREPE
* Provides information on which notes were out of tune
* Outputs practice tips based on playing
* Graphs played frequencies compared to target frequencies

## Requirements
* Python 3.9+
* crepe, librosa, numpy, matplotlib.pyplot, converter (from music21), os, logging, sys, google.generativeai

## Installation
1. Clone the repository

```
bash
git clone https://github.com/yourusername/violin-intonation.git
cd violin-intonation
```

2. Install dependencies using requirements.txt
`pip install -r requirements.txt`

## Setup
1. Create a `.env` file in the project root to store your Gemini API key

    `GEMINI_API_KEY=your_api_key_here`

2. Add `.env` to `.gitignore` so it is not pushed to GitHub:

    `.env`

## Usage
Run the program locally with: `node server.js`

Then, open the provided URL (usually http://localhost:3000/) in your browser.
1) Choose your sheet music (.pdf) and violin recording (.wav) files
2) Type in the tempo at which you played (in BPM)
2) Press "Upload"

## Limitations
* Currently supports local hosting only
* Solo violin
* Need to have accurate rhythm and tempo
* Works best in quiet environments
