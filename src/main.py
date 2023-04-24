import os
from data_preprocessing import audio_conversion, feature_extraction, tab_tokenization, data_split
from models import seq2seq, train
from evaluation import evaluation_metrics

def main():
    # Data preparation
    print("audio_conversion start")
    audio_conversion.convert_audio_to_midi("data/raw_audio", "data/midi")
    print("feature_extraction start")
    feature_extraction.extract_audio_features("data/midi", "data/preprocessed/audio_features")
    print("tab_tokenization start")
    tab_tokenization.tokenize_tabs("data/tabs", "data/preprocessed/tokenized_tabs")
    
    # Data split
    print("split_data start")
    data_split.split_data("data/preprocessed", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    # Model definition
    model = seq2seq.create_model()
    
    # Model training
    print("train start")
    train.train_model()
    
    # Model evaluation
    print("eval start")
    test_score = evaluation_metrics.evaluate_model(model, "data/preprocessed/test")
    print("Test Score:", test_score)

if __name__ == "__main__":
    main()
