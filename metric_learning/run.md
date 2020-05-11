python experiment_siamese.py --experiment_folder='result_siamese_dim16' --embed=16 --batch_size=64 > result_siamese_dim16.log 2>&1 &


python experiment_siamese.py --experiment_folder='result_siamese_dim32' --embed=32 --batch_size=64 > result_siamese_dim32.log 2>&1 &

python experiment_siamese.py --experiment_folder='result_siamese_dim2' --embed=2 --batch_size=64 > result_siamese_dim2.log 2>&1 &


python experiment_triplet.py > result_triplet_dim2.log 2>&1 &

