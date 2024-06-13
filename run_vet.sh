##################### vet data - no filter #####################
# experiment 번호의 아래 코드를 terminal 창에 붙여넣고 재생하면 됩니다
# ex: 1번 -> python3.8 vet.py --semi_method mix --gpu 2 --dataset vet --ratio 4 --num_max 500 --imb_ratio_l 100 --imb_ratio_u 1 --epochs 100 --val-iteration 771 --batch-size 8

# Mixmatch
# 1번
#python3.8 vet.py --semi_method mix --gpu 2 --dataset vet --ratio 4 --num_max 500 --imb_ratio_l 100 --imb_ratio_u 1 --epochs 100 --val-iteration 771 --batch-size 8
# Mixmatch + DARP
# 2번
python3.8 vet.py --semi_method mix --gpu 2 --dataset vet --darp --align --alpha 2 --warm 250 --ratio 4 --num_max 500 --imb_ratio_l 100 --imb_ratio_u 1 --epochs 100 --val-iteration 771 --batch-size 8

# Remixmatch
# 3번
#python3.8 vet.py --semi_method remix --gpu 2 --dataset vet --align --ratio 4 --num_max 500 --imb_ratio_l 100 --imb_ratio_u 1 --epochs 100 --val-iteration 771 --batch-size 8
# Remixmatch + DARP
# 4번
#python3.8 vet.py --semi_method remix --gpu 2 --dataset vet --darp --align --alpha 2 --warm 250 --ratio 4 --num_max 500 --imb_ratio_l 100 --imb_ratio_u 1 --epochs 100 --val-iteration 771 --batch-size 8

# Fixmatch
# 5번
#python3.8 vet.py --semi_method fix --gpu 2 --dataset vet --align --ratio 4 --num_max 500 --imb_ratio_l 100 --imb_ratio_u 1 --epochs 100 --val-iteration 771 --batch-size 8
# Fixmatch+DARP
# 6번
#python3.8 vet.py --semi_method fix --gpu 2 --dataset vet --darp --align --alpha 2 --warm 250 --ratio 4 --num_max 500 --imb_ratio_l 100 --imb_ratio_u 1 --epochs 100 --val-iteration 771 --batch-size 8

##################### vet data - yes filter #####################

# Remixmatch - ours
# 7번
#python3.8 vet_ours.py --change_aug yes --semi_method remix --gpu 2 --dataset vet --align --ratio 4 --num_max 500 --imb_ratio_l 100 --imb_ratio_u 1 --epochs 100 --val-iteration 771 --batch-size 8
# Remixmatch+DARP - ours
# 8번
#python3.8 vet_ours.py --change_aug yes --semi_method remix --gpu 2 --dataset vet --darp --align --alpha 2 --warm 250 --ratio 4 --num_max 500 --imb_ratio_l 100 --imb_ratio_u 1 --epochs 100 --val-iteration 771 --batch-size 8

# Fixmatch - ours
# 9번
#python3.8 vet_ours.py --change_aug yes --semi_method fix --gpu 2 --dataset vet --align --ratio 4 --num_max 500 --imb_ratio_l 100 --imb_ratio_u 1 --epochs 150 --val-iteration 771 --batch-size 8
# Fixmatch+DARP - ours
# 10번
#python3.8 vet_ours.py --change_aug yes --semi_method fix --gpu 2 --dataset vet --darp --align --alpha 2 --warm 250 --ratio 4 --num_max 500 --imb_ratio_l 100 --imb_ratio_u 1 --epochs 150 --val-iteration 771 --batch-size 8
