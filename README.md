# violin-teacher

This project detects intonation errors in violin recordings. It analyzes input audio (from a recorded file) and provides feedback on how sharp or flat the played pitch is compared to the target frequency, as well as tips on how the user can practice. 

Link to research paper: https://dpl6hyzg28thp.cloudfront.net/media/Learning_to_Listen__Machine_Learning_for_Intonation_Error_Detection_in_Violin_Performance.pdf

---

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Notes](#limitations--notes)
- [License](#license)

---

## Features
* Pitch detection using CREPE
* Provides information on which notes were out of tune
* Outputs practice tips based on playing
* Graphs played frequencies compared to target frequencies

## Requirements
* Python 3.9+
* Python packages: crepe, librosa, numpy, matplotlib, music21, scipy, python-dotenv, google-generativeai
* Java
* Audiveris

## Installation
1. Clone the repository

    ```
    bash
    git clone https://github.com/isabelle248/polygence-website.git
    cd polygence-website
    ```

2. Install python packages using requirements.txt

    `pip install -r requirements.txt`

3. Install non-Python tools

    * Audiveris — for converting PDF sheet music to MusicXML (.mxl):
        * Download: https://audiveris.github.io/
        * Requires Java: https://www.java.com/
        * Test installation:

            `/Applications/Audiveris.app/Contents/MacOS/Audiveris -help`

    * CREPE dependencies: Make sure TensorFlow is installed if not included automatically:

        `pip install tensorflow`


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

<img width="524" height="312" alt="image" src="https://github.com/user-attachments/assets/e608e142-9c3e-4201-bc28-c1ef66a50b78" />

## File Structure

```
violin-teacher/
│
├── public/
│   ├── index.html          # Main HTML page
├── scripts/
│   ├── polygence.sh        # Script for PDF processing
│   ├── run_crepe.py        # Script for audio pitch analysis
├── uploads/            # Folder to store uploaded files
├── server.js           # Node server
├── setup.sh            # Setup script
├── package.json        # Node dependencies
├── package-lock.json   # Node lock file
├── requirements.txt    # Python dependencies
├── .gitignore          # Files/folders to ignore in git
├── README.md           # This file
```

## Notes
* Currently supports local hosting only
* Solo violin
* Need to have accurate rhythm and tempo
* Works best in quiet environments
