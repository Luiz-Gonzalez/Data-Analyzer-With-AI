# Data-Analyzer-With-AI

Voice-based agent for **data analysis** using **Python, LangChain, Whisper, and OpenAI APIs**.  
This project lets you ask questions by voice, transcribe them to text, process them with a Large Language Model (LLM), and receive spoken answers based on your dataset.

---

## Features

1. Voice input controlled by a hotkey  
2. Speech-to-Text transcription with Whisper  
3. AI-powered data analysis with Pandas DataFrame  
4. Spoken response using OpenAI Text-to-Speech  

---

## Requirements

- Python 3.11.9  
- Dependencies listed in `requirements_luiz.txt`  
- OpenAI API key (set in a `.env` file)  
- A CSV file for analysis (default: `df_pets.csv`, but you can replace it with any other file by changing the code)  

---

## Installation

- Clone this repository  
- Create and activate a virtual environment  
- Install dependencies from `requirements_luiz.txt`  
- Add your OpenAI API key to a `.env` file  

---

## Usage

- Place your CSV file in the project folder  
- Update the code if you want to use a different filename  
- Run the program  
- Press the hotkey to start and stop voice recording  
- Ask your questions about the dataset  
- The system will transcribe your voice, analyze the data, and read the answer back to you  

---

## Example

You say: “Show me only the rows where species is cat.”  
The AI responds: “Here are the rows where the species is cat.”  

---
