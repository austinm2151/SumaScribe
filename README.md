# SumaScribe

This project processes and summarizes text and audio files using a pre-trained Llama2 model and Whisper speech-to-text technology. It handles multiple input formats and organizes summaries into a structured Word document.

## Getting Started

Follow these steps to set up the project and start processing your files.

### Prerequisites

- Install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- Install [Visual Studio Code](https://code.visualstudio.com/) or any other preferred IDE.

### Installation Steps

1. **Clone the Repository:**
   Copy the GitHub repository directory to your local machine.

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Open the Project in Your IDE:**
   - Launch your IDE (e.g., VS Code).
   - Open the project folder in the IDE.

3. **Create a Conda Environment:**
   - Ensure you have Conda installed.
   - Create the environment using the `environment.yml` file:

     ```bash
     conda env create -f environment.yml
     ```

   - Activate the environment:

     ```bash
     conda activate <environment-name>
     ```

     Replace `<environment-name>` with the name specified in the `environment.yml` file.

4. **Place Input Files:**
   - **Audio Files:** Place `.m4a`, `.mp3`, `.mp4`, or `.wav` files in the `Audio_Inputs` folder.
   - **Text Files:** Place `.txt`, `.pdf`, or `.docx` files in the `Text_Inputs` folder.

5. **Run the Project:**
   - Start the main script by running `start.py`:

     ```bash
     python start.py
     ```

### Folder Structure

Ensure the folder structure is as follows:

```
project-folder/
├── Audio_Inputs/      # Place audio files here
├── Audio_Outputs/     # Processed audio files will be stored here
├── Text_Inputs/       # Place text files here
├── Note_Outputs/      # Summarized notes will be saved here
├── environment.yml    # Environment configuration file
├── start.py           # Main script to run the project
├── README.md          # Instructions (this file)
```

### Usage Notes

- Ensure the inputs are placed in the appropriate folders (`Audio_Inputs` for audio files, `Text_Inputs` for text files).
- The processed outputs will be saved in the corresponding output folders (`Audio_Outputs` or `Note_Outputs`).
- If the process is interrupted using the `Esc` key, partial progress will be saved, and you can resume without reprocessing completed files.

### Supported Formats

- **Audio Inputs:** `.m4a`, `.mp3`, `.mp4`, `.wav`
- **Text Inputs:** `.pdf`, `.docx`, `.txt`

### Features

- Converts audio to text using the Whisper model.
- Cleans and structures text files.
- Summarizes processed text and organizes results in a Word document.

### Troubleshooting

- **Missing Dependencies:** Ensure you activated the correct Conda environment and installed all dependencies listed in `environment.yml`.

- **Invalid File Formats:** Use only the supported file formats for input:
  - **Audio Inputs:** `.m4a`, `.mp3`, `.mp4`, `.wav`
  - **Text Inputs:** `.pdf`, `.docx`, `.txt`

- **Folder Missing:** Ensure all required folders (`Audio_Inputs`, `Text_Inputs`, etc.) exist in the project directory. If not, create them manually.

- **First-Time Model Download:** The first time the Llama2 or Whisper model is used, it may take some time to download as the models are fetched from online repositories. Ensure your internet connection is stable during this process.

- **GPU Requirements:** Ensure your system has a GPU with at least **12GB of VRAM** to handle the Llama2 model efficiently. If your system has insufficient GPU memory, the process may fail or slow down significantly. Consider reducing model size or running on a system with higher specifications.

- **Performance Issues:** If the process is slow or freezes:
  - Check that the appropriate hardware resources (GPU/CPU) are being utilized.
  - Ensure the input files are correctly placed in the designated folders.

- **Unexpected Errors:** If you encounter an unknown issue, review the console output for error messages and consult the project documentation or source code comments for guidance.

---

Enjoy processing and summarizing your files!
