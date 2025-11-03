"""Inference script for making predictions with trained model."""

import argparse
import torch
from pathlib import Path

from .encoder import LegalBERTEncoder
from .model import DeepResidualMLP


class ClaudetteClassifier:
    """Wrapper for trained Claudette classifier."""

    def __init__(self, model_path: str, device: str = "auto"):
        """Load trained model.

        Args:
            model_path: Path to saved model checkpoint
            device: Device to run on (auto, cuda, mps, cpu)
        """
        self.device = self._get_device(device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Get config
        config = checkpoint['config']

        # Initialize encoder
        self.encoder = LegalBERTEncoder(
            model_name=config.encoder_name,
            freeze_encoder=False
        ).to(self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])

        # Initialize classifier
        embedding_dim = self.encoder.get_embedding_dim()
        self.classifier = DeepResidualMLP(
            input_dim=embedding_dim,
            hidden_dims=config.hidden_dims,
            num_residual_blocks=config.num_residual_blocks,
            dropout=config.dropout
        ).to(self.device)
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])

        # Set to eval mode
        self.encoder.eval()
        self.classifier.eval()

        print(f"Loaded model from {model_path}")
        print(f"Device: {self.device}")

    def _get_device(self, device_str: str) -> torch.device:
        """Get PyTorch device."""
        if device_str == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device_str)
        return device

    def predict(self, texts: list[str]) -> list[dict]:
        """Predict fairness for given texts.

        Args:
            texts: List of clause texts

        Returns:
            List of predictions with probabilities
        """
        with torch.no_grad():
            # Get embeddings
            embeddings = self.encoder(texts, self.device)

            # Get predictions
            logits = self.classifier(embeddings)
            probs = torch.sigmoid(logits).squeeze(1)

            # Convert to predictions
            predictions = []
            for i, text in enumerate(texts):
                prob_unfair = probs[i].item()
                prob_fair = 1 - prob_unfair
                pred_label = "unfair" if prob_unfair > 0.5 else "fair"

                predictions.append({
                    'text': text,
                    'prediction': pred_label,
                    'probability_fair': prob_fair,
                    'probability_unfair': prob_unfair,
                    'confidence': max(prob_fair, prob_unfair)
                })

            return predictions

    def predict_single(self, text: str) -> dict:
        """Predict fairness for a single text.

        Args:
            text: Clause text

        Returns:
            Prediction dictionary
        """
        return self.predict([text])[0]


def main():
    """CLI for inference."""
    parser = argparse.ArgumentParser(description="Claudette classifier inference")
    parser.add_argument('--model', type=str,
                       default='results/claudette_classifier/best_model.pt',
                       help='Path to trained model')
    parser.add_argument('--text', type=str, help='Single clause text to classify')
    parser.add_argument('--file', type=str, help='File with clauses (one per line)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'])

    args = parser.parse_args()

    # Load model
    classifier = ClaudetteClassifier(args.model, device=args.device)

    # Get texts
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # Interactive mode
        print("\nInteractive mode. Enter clauses (empty line to quit):")
        texts = []
        while True:
            text = input("\n> ")
            if not text:
                break
            texts.append(text)

    if not texts:
        print("No texts provided")
        return

    # Make predictions
    predictions = classifier.predict(texts)

    # Print results
    print("\n" + "=" * 80)
    print("Predictions")
    print("=" * 80)

    for i, pred in enumerate(predictions, 1):
        print(f"\n[{i}] {pred['text'][:100]}{'...' if len(pred['text']) > 100 else ''}")
        print(f"Prediction: {pred['prediction'].upper()}")
        print(f"Confidence: {pred['confidence']:.2%}")
        print(f"P(fair)={pred['probability_fair']:.4f}, P(unfair)={pred['probability_unfair']:.4f}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
