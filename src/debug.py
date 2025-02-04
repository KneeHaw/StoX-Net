import torch
import torch.nn.functional as F
import os
from matplotlib import pyplot as plt
import numpy as np

subarray_dimension = 256

log1 = './raw_tens.txt'
log2 = './norm_tens.txt'


def update_txt(tensor_str, option):
    if option == log1:
        with open(log1, 'a') as f:
            f.write(tensor_str)
    elif option == log2:
        with open(log2, 'a') as f:
            f.write(tensor_str)


def update_pt(tensor, path):
    if os.path.getsize(path) > 0:
        existing = [torch.load(path)]
        save_list = existing.append(tensor.unsqueeze(-1))
        torch.save(save_list, path)
    else:
        torch.save([tensor.unsqueeze(-1)], path)


def tensor_stats(tensor):
    min = torch.min(tensor).item()
    max = torch.max(tensor).item()
    mean = torch.mean(tensor).item()
    std_dev = torch.std(tensor).item()
    out_str = f"{min:.3f}, {max:.3f}, {mean:.3f}, {std_dev:.3f}"
    print(f"\t\tTensor Stats: {out_str}")


def plot_tensor_hist(tensor):
  plt.hist(tensor.detach().cpu().flatten().squeeze().numpy(), bins=100)
  plt.show()


def print_num_unique_vals(tensor):
  unique_values, counts = torch.unique(tensor, return_counts=True)
  num_distinct_values = len(unique_values)
  print(f"\t\tDistinct Values: {num_distinct_values}")


def plot_tensor_plot(tensorx, tensory):
    plt.scatter(tensorx.detach().cpu().flatten().squeeze().numpy(), tensory.detach().cpu().flatten().squeeze().numpy())
    plt.show()


def save_tensor_csv_raw_norm(raw_tensor, norm_tensor, name):
    file_name = name[:-3]
    print("saving data to file")
    raw_tensor1 = raw_tensor.detach().cpu().flatten().squeeze().numpy()
    norm_tensor1 = norm_tensor.detach().cpu().flatten().squeeze().numpy()
    raw_file = open(file_name + '_raw.dat', 'ab')
    norm_file = open(file_name + '_norm.dat', 'ab')
    np.savetxt(raw_file, raw_tensor1)
    np.savetxt(norm_file, norm_tensor1)
    raw_file.close()
    norm_file.close()


# if __name__ == '__main__':
#     values = [-1, 1]
#     num_steps = 4

    # -1, -1, -1, -1
    # -1, -1, -1,  1
    # -1, -1,  1,  1
    # -1,  1,  1,  1
    #  1,  1,  1,  1

    # -1
    # -0.5
    # 0
    # 0.5
    # 1

    # -1, -1, -1
    # -1, -1,  1
    # -1,  1,  1
    #  1,  1,  1

    # -1
    # -1/3
    #  1/3
    #  1

    # for i in range(num_steps ** 2):
    #     step_result = [values[0] for x ]


