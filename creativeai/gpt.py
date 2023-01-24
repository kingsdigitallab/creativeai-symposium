import argparse
import os

import torch
from happytransformer import GENSettings, GENTrainArgs, HappyGeneration


class Generator:
    def __init__(self, model_name="GPT-NEO", model_path="EleutherAI/gpt-neo-1.3B"):
        self._model_name = model_name
        self._model_path = model_path

        if os.path.exists(self._model_path):
            self.model = HappyGeneration(load_path=self._model_path)
        else:
            self.model = HappyGeneration(self._model_name, self._model_path)

    def generate(
        self,
        prompt: str,
        max_length: int = 200,
        do_sample: bool = True,
        top_k: int = 50,
        temperature: float = 0.5,
        no_repeat_ngram_size: int = 2,
    ) -> str:
        args = GENSettings(
            max_length=max_length,
            do_sample=do_sample,
            early_stopping=True,
            top_k=top_k,
            temperature=temperature,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        result = self.model.generate_text(prompt, args=args)

        return result.text

    def train(self, data_path: str, label: str, epochs: int = 3):
        self.model.train(data_path, args=GENTrainArgs(num_train_epochs=epochs))

        self.model.save(label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train a GPT model")
    parser.add_argument("model", default="EleutherAI/gpt-neo-1.3B", help="Model path")
    parser.add_argument("data", help="Text file with training data")
    parser.add_argument("-n", "--name", default="GPT", help="Model name")
    parser.add_argument(
        "-o",
        "--output",
        default="data/final/trained",
        help="Path to save the trained model",
    )
    parser.add_argument("-e", "--epochs", default=4)
    args = parser.parse_args()

    g = Generator(model_name=args.name, model_path=args.model)
    g.train(args.data, args.output, args.epochs)
