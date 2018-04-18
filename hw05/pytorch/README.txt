# Create customized embeddings file and data set.
python3 pytorch/data_set.py snli/snli_1.0 snli/GoogleNews-vectors-negative300.bin snli/embedding.pkl snli/snli_padding.pkl

# Run experiment with the data set created above.
python3 pytorch/rnnExercise.py <mode>


