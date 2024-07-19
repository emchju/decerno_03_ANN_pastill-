glodbal_settings = {
  "test_size": 0.2,
  "random_state": 42, # clf = 0 / tkn = 42
  "network": 'tkn', # clf, tkn
  "vocab_size": 50000,
  "embedding_dim": 96,
  "max_length": 2000,
  "epochs": 100,
  "batch_size": 32,
  "data_path": '../saved_data/dataset_encoded.csv',
  "model_path_ann": '../saved_model/ANN_model',
  "model_path_rnn": '../saved_model/RNN_model_',
  "model_path_transformer": '../saved_model/transformer_model_',
  "scaler_path": '../saved_scaler/scaler.sav',
}