# AI Storybook Backend Thing

This is the backend server for my AI storybook project.

## What it Does

Basically, you run this Python script, and it starts a local web server (using Gradio). You can then go to that webpage in your browser, upload a picture of a face, and type a text prompt (like "knight in armor" or "astronaut on mars").

The script takes the prompt, generates an image using Stable Diffusion, finds the face in your uploaded picture and the generated picture, swaps them using Insightface, and then tries to clean up the swapped face a bit using GFPGAN.

The final image with the swapped face is shown on the webpage.

It all runs on your own computer.

## How it Works (Tech Stuff)

* **Python:** The main language.
* **Gradio:** Creates the simple web interface and API.
* **Diffusers (Hugging Face):** Runs Stable Diffusion 1.5 to make the pictures from text.
* **Insightface:** Does the face detection and the face swapping part (uses an ONNX model).
* **GFPGAN:** Tries to make the swapped face look better.
* **PyTorch:** Needed for Stable Diffusion and GFPGAN.
* **ONNX Runtime:** Needed for Insightface.
* **Git LFS:** Used because the `inswapper_128.onnx` file is huge (like 500MB) and GitHub doesn't like big files directly.

## How to Run It

1.  **Clone:** Get the code from GitHub:
    ```bash
    git clone [https://github.com/Subbbhash/ai-storybook-backend.git](https://github.com/Subbbhash/ai-storybook-backend.git)
    cd ai-storybook-backend
    ```
2.  **Install Git LFS:** You NEED Git LFS installed for the big model file. Download it from [git-lfs.com](https://git-lfs.com/) and run `git lfs install` once on your machine if you haven't before.
3.  **Pull LFS File:** After cloning, make sure you get the actual model file:
    ```bash
    git lfs pull
    ```
4.  **Python:** Make sure you have Python installed (I used Python 3.12, maybe 3.10+ works?).
5.  **Virtual Environment:** It's best to use one:
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On Mac/Linux:
    # source venv/bin/activate 
    ```
6.  **Install Libraries:** Install everything needed:
    ```bash
    pip install -r requirements.txt
    ```
    * **WINDOWS USERS:** The `insightface` install might fail if you don't have Microsoft C++ Build Tools. You need to install "Build Tools for Visual Studio" and make sure the "Desktop development with C++" workload is selected during its install. Then try `pip install` again.
    * **GPU:** The `requirements.txt` might install GPU versions (`onnxruntime-gpu`, `torch` with CUDA). Make sure you have NVIDIA drivers installed if you want it to use your GPU (it's WAY faster). If you don't have an NVIDIA GPU, you might need to edit `requirements.txt` to install `onnxruntime` (CPU version) and the CPU version of `torch`. The code tries to use CUDA if available.
7.  **Run the Server:**
    ```bash
    python app.py
    ```
8.  **Open in Browser:** The terminal will show a URL like `http://127.0.0.1:7860`. Open that in your web browser.
9.  **Use:** Upload a face, type prompts, click generate. Wait for it (GPU = seconds, CPU = many minutes).

## Notes

* This runs **locally**. It uses your computer's resources (CPU/GPU/RAM).
* **GPU highly recommended**. It's painfully slow on just CPU.
* Quality is okay-ish. Depends a lot on the prompts you give and how well the generated face matches the pose/lighting of your uploaded face. GFPGAN helps but isn't magic.
* Sometimes the face detection fails (on source or generated image), it will show an error. Try a different picture or prompt.
