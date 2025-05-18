

ENV_1="dip-env-rnn"
ENV_2="dip-env-vae"
CURRENT_ENV="$CONDA_DEFAULT_ENV"
if [ "$CURRENT_ENV" != "$ENV_1" ]; then
	echo "Switching to $ENV_1 environment..."
	conda deactivate
	conda activate $ENV_1
else
	echo "Switching to $ENV_2 environment..."
	conda deactivate
	conda activate $ENV_2
fi