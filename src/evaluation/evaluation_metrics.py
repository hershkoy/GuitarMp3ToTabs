import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from python_Levenshtein import distance as levenshtein_distance


def bleu_score(reference, candidate):
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    smoothing = SmoothingFunction().method1
    return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)


def edit_distance(reference, candidate):
    return levenshtein_distance(reference, candidate)


def evaluate_model(model, test_data, metric="bleu"):
    assert metric in ["bleu", "edit"], "Invalid metric"

    scores = []

    for input_features, ground_truth in test_data:
        # Generate prediction using the model
        predicted_output = model.generate(input_features)
        
        # Calculate the evaluation metric
        if metric == "bleu":
            score = bleu_score(ground_truth, predicted_output)
        elif metric == "edit":
            score = edit_distance(ground_truth, predicted_output)

        scores.append(score)

    # Calculate the average score
    return np.mean(scores)


if __name__ == "__main__":
    # Replace the following with your actual test data and trained model
    test_data = [("input_features_1", "ground_truth_1"), ("input_features_2", "ground_truth_2")]
    model = None  # Replace with your trained model instance

    bleu_test_score = evaluate_model(model, test_data, metric="bleu")
    print(f"Average BLEU score: {bleu_test_score}")

    edit_test_score = evaluate_model(model, test_data, metric="edit")
    print(f"Average edit distance: {edit_test_score}")
