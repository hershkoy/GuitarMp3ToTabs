import os
from data_preprocessing import audio_conversion, feature_extraction, tab_tokenization, data_split
from models import seq2seq, train
from evaluation import evaluation_metrics

def main():
    # Data preparation
    audio_conversion.convert_audio_to_midi("data/raw_audio", "data/midi")
    feature_extraction.extract_audio_features("data/midi", "data/preprocessed/audio_features")
    tab_tokenization.tokenize_tabs("data/tabs", "data/preprocessed/tokenized_tabs")
    
    # Data split
    data_split.split_data("data/preprocessed", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    # Model definition
    model = seq2seq.create_model()
    
    # Model training
    train.train_model(model, "data/preprocessed")
    
    # Model evaluation
    test_score = evaluation_metrics.evaluate_model(model, "data/preprocessed/test")
    print("Test Score:", test_score)

if __name__ == "__main__":
    main()
