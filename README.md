# Creative AI: Theory and Practice

This repository contains experimental code used for the KDL presentation,
_From Once Upon a Time, to Happily Ever After, via AI_ at the
[Creative AI: Theory and Practice symposium](https://www.kcl.ac.uk/ai/assets/pdf/creative-ai-programme-270123.pdf).

## Set up

Install [poetry](https://python-poetry.org/docs/#installation) and the requirements:

    poetry install

## Fine tune the text generation model

    poetry run python creativeai/gpt.py MODEL_PATH DATA_PATH

- `MODEL_PATH` can either be a [Hugging Face](https://huggingface.co/) path or a local path.
- `DATA_PATH` should be a path to a text file with all the training data.

To see a list of all the available options for the text generation run the `gpt.py`
script with the `--help` option:

    poetry run python creativeai/gpt.py --help

    usage: gpt.py [-h] [-n NAME] [-o OUTPUT] [-e EPOCHS] model data

    Script to train a GPT model

    positional arguments:
      model                 Model path
      data                  Text file with training data

    optional arguments:
      -h, --help            show this help message and exit
      -n NAME, --name NAME  Model name
      -o OUTPUT, --output OUTPUT
                            Path to save the trained model
      -e EPOCHS, --epochs EPOCHS

> **Warning**: Depending on the model being used, fine tunning a text generation model
> can be very time consuming without access to a GPU.

## Run the notebook

The repository also contains a [notebook](notebooks/creativeai.ipynb) with an interface
to generate text and imgages from the generated text. The [story](story.pdf) document
contains an example of a story generated in the notebook for the prompt:
_There was a man made of clockwork who longed to become human so_, with the settings
_top p_ set to 0.9, and _ngrams_ set to 4.

> **Warning**: The notebook is set up to use GPUs.

## Data

The models for the presentation were fine tuned using the
[Grimm's Fairy Tales](https://www.gutenberg.org/files/2591/2591-0.txt).

## Models

- [GPT-Neo 1.3B](https://huggingface.co/EleutherAI/gpt-neo-1.3B) for text generation;
- [Stable diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1) for image
  generation.

<div class="csl-bib-body">
  <div data-csl-entry-id="gpt-neo" class="csl-entry">Black, S., Leo, G., Wang, P., Leahy, C., &#38; Biderman, S. (2021). <i>GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow</i> (1.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.5297715</div>
  <div data-csl-entry-id="gao2020pile" class="csl-entry">Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N., &#38; others. (2020). The Pile: An 800GB Dataset of Diverse Text for Language Modeling. <i>ArXiv Preprint ArXiv:2101.00027</i>.</div>
  <div data-csl-entry-id="Rombach_2022_CVPR" class="csl-entry">Rombach, R., Blattmann, A., Lorenz, D., Esser, P., &#38; Ommer, B. (2022). High-Resolution Image Synthesis With Latent Diffusion Models. <i>Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)</i>, 10684–10695.</div>
</div>
