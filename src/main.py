import os
from data_preprocessing import audio_conversion, feature_extraction, tab_tokenization, data_split
from models import seq2seq, train
from evaluation import evaluation_metrics

def main():

    project_root = os.path.dirname(os.path.abspath(__file__))

    print("audio_conversion start")
    audio_input_dir = os.path.join(project_root,'..', 'data', 'raw_audio')  
    audio_output_dir = os.path.join(project_root,'..', 'data', 'preprocessed', 'audio_features')  
    time_signature = (6, 8)   
    audio_conversion.convert_all_audio_files(audio_input_dir, audio_output_dir, time_signature)

    print("feature_extraction start")
    feature_input_dir = os.path.join(project_root,'..', 'data', 'midi') 
    feature_output_dir = os.path.join(project_root,'..', 'data', 'preprocessed', 'audio_features')     
    feature_extraction.extract_audio_features(feature_input_dir, feature_output_dir)

    print("tab_tokenization start")
    tokens_input_dir = os.path.join(project_root,'..', 'data', 'tabs')
    tokens_output_dir = os.path.join(project_root,'..', 'data', 'preprocessed', 'tokenized_tabs') 
    tokenizer_path = "tab_tokenizer.npz"
    tab_tokenization.tokenize_tabs(tokens_input_dir, tokens_output_dir, tokenizer_path) 
    
    # Data split
    print("split_data start")
    datasplit_audio_input_dir = os.path.join(project_root,'..', 'data', 'preprocessed', 'audio_features')  
    datasplit_tab_input_dir = os.path.join(project_root,'..', 'data', 'preprocessed', 'tokenized_tabs')
    datasplit_output_dir = os.path.join(project_root,'..', 'data', 'preprocessed', 'synchronized')    
    data_split.synchronize_and_split_data(datasplit_audio_input_dir, datasplit_tab_input_dir, datasplit_output_dir)    
    
    # Model definition
    input_dim = 13  # Replace with the dimension of your input features (e.g., MFCCs)
    output_dim = 100  # Replace with the size of your tokenized tab vocabulary
    latent_dim = 256
    model = seq2seq.create_seq2seq_model(input_dim, output_dim, latent_dim)   
    
    # Model training
    print("train start")
    train.train_model(model)
    
    # Model evaluation
    print("eval start")
    test_audio_features_dir = os.path.join(project_root,'data', 'preprocessed', 'audio_features', 'test')
    test_tokenized_tabs_dir = os.path.join(project_root,'data', 'preprocessed', 'tokenized_tabs', 'test')
    test_score = evaluation_metrics.evaluate_model(model, test_audio_features_dir, test_tokenized_tabs_dir)
    print("Test Score:", test_score)

if __name__ == "__main__":
    main()
