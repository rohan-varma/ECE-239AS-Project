- Encoder decoder model (LSTM)
- Encoder: use pretraining to just predict itself. i.e. input time series, encoder just wants to predict the input time series
- use these pretrained weights to get fixed-length representation
- fixed length representation can be input into a decoder.
- decoder can:
	- predict action
	- generate additional timesteps
- pretraining with encoder, & dataset augmentation maybe?


- Vanilla RNN/LSTM to just predict the next action
- RNN/lstm to predict a sequence of actions
- given some electrodes at timesteps, predict the next few timesteps
- use encoder/decoder as a feature extractor
- bidirectional RNN
