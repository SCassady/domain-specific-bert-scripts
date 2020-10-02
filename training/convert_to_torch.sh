# Used to convert pretrained tensorflow bert models for use with pytorch finetuning scripts.
# More information at: https://huggingface.co/transformers/v2.4.0/converting_tensorflow_models.html

export NCBI_MODEL_DIR="/disk/data2/radiology/users/scassady/ncbi"
export DATA_DIR="/disk/data2/radiology/users/scassady/data"
export BERT_BASE_DIR="/disk/data2/radiology/users/scassady/bert-BASE"

transformers-cli convert --model_type bert --tf_checkpoint $DATA_DIR/model.ckpt-12000 --config $NCBI_MODEL_DIR/bert_config.json --pytorch_dump_output $DATA_DIR/pytorch_model.bin
echo "--------------------------- conversion complete -----------------------------"
