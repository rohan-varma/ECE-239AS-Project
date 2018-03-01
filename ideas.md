- Encoder decoder model (LSTM)
- Encoder: use pretraining to just predict itself. i.e. input time series, encoder just wants to predict the input time series
- use these pretrained weights to get fixed-length representation
- fixed length representation can be input into a decoder.
- decoder can:
	- predict action
	- generate additional timesteps
- pretraining with encoder, & dataset augmentation maybe?

"Given a univariate time series , the encoder LSTM reads in the first T timestamps , and constructs a fixed-dimensional embedding state. From the embedded state, the decoder LSTM then constructs the following F timestamps , which are also guided via  (as showcased in the bottom panel of Figure 1). In order to construct the next few time steps from the embedding, it must contain the most representative and meaningful aspects from the input time series."

- Vanilla RNN/LSTM to just predict the next action
- RNN/lstm to predict a sequence of actions
- given some electrodes at timesteps, predict the next few timesteps
- use encoder/decoder as a feature extractor