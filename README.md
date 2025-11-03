# polygence-website

## Overview

This project detects intonation errors in violin recordings. It analyzes input audio (from a recorded file) and provides feedback on how sharp or flat the played pitch is compared to the target frequency, as well as tips on how the user can practice.

## Features
* Pitch detection using CREPE
* Provides information on which notes were out of tune
* Outputs practice tips based on playing
* Graphs played frequencies compared to target frequencies

## Installation
Run the program locally with: `node server.js`

Then, open the provided URL (usually http://localhost:3000/) in your browser.

## Requirements
* Python 3.9+
* crepe, librosa, numpy, matplotlib.pyplot, converter (from music21), os, logging, sys, google.generativeai

## Usage
1) Choose your sheet music (.pdf) and violin recording (.wav) files
2) Type in the tempo at which you played (in BPM)
2) Press "Upload"

## Limitations
* Currently supports local hosting only
* Solo violin
* Need to have accurate rhythm and tempo
* Works best in quiet environments
