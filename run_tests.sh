# Run GA training
cd hp_optimization
python3 main.py

# Run regular training
cd ..
python3 train.py -data data/demo -save_model saved_models/regular -gpu_ranks 0 --batch_size 192 --valid_steps 144 --valid_batch_size 192 --train_steps 14400 --optim adam --label_smoothing 0.1 --learning_rate 0.001 --learning_rate_decay 0.1 --start_decay_steps 0 --decay_steps 2880 --report_every 144 --log_file saved_models/regular.log
