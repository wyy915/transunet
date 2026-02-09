import os
import random
import os

# split train and test cases
origin_dir = "./origin"
list_dir = "../../lists/lists_knee"
os.makedirs(list_dir, exist_ok=True)
cases = sorted(os.listdir(origin_dir))
all_list_path = os.path.join(list_dir, "all_list.txt")
with open(all_list_path, "w") as f:
    for case in cases:
        f.write(f"{case}.npy.h5\n")

print(f"✔ 已生成 all_list.txt，共 {len(cases)} 个 case")

random.seed(1234)
with open(all_list_path) as f:
    all_cases = [line.strip() for line in f.readlines()]

random.shuffle(all_cases)

num_cases = len(all_cases)
num_train = int(0.8 * num_cases)

train_cases = all_cases[:num_train]
test_cases = all_cases[num_train:]

with open(os.path.join(list_dir, "train.txt"), "w") as f:
    for c in train_cases:
        f.write(c + "\n")

with open(os.path.join(list_dir, "test_vol.txt"), "w") as f:
    for c in test_cases:
        f.write(c + "\n")

print(f"✔ train cases: {len(train_cases)}")
print(f"✔ test cases: {len(test_cases)}")
