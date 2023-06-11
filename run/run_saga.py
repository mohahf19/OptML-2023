# batch_size_full = 10**20
# num_steps = 20000
# num_parts = 10
#
# # make data loader for testing
# test_loader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=batch_size_full, shuffle=True
# )
#
# # make data loaders for training
# perm = torch.randperm(len(train_dataset)).tolist()
# shuffled_dataset = torch.utils.data.Subset(train_dataset, perm)
#
# train_loader = torch.utils.data.DataLoader(
#     shuffled_dataset, batch_size=1, shuffle=True
# )
#
# train_loader_temp = torch.utils.data.DataLoader(
#     shuffled_dataset, batch_size=len(train_dataset) // num_parts, shuffle=False
# )
#
# assignment = [-1 for _ in range(len(train_dataset))]
# batches = []
# p = 0
# for data, targets, indices in train_loader_temp:
#     batches.append(indices)
#     print(indices)
#     for j in indices:
#         assignment[j] = p
#     p += 1
# assert len([p for p in assignment if p == -1]) == 0
#
# s = len(train_dataset) // num_parts
# train_loder_partitions = []
# for p in range(num_parts):
#     B = torch.utils.data.Subset(shuffled_dataset, [p * s + x for x in range(s)])
#     train_loder_partitions.append(
#         torch.utils.data.DataLoader(B, batch_size=s, shuffle=False)
#     )
#
# network = NN()
# network.to(device)
#
# network_temp = []
# for p in range(num_parts):
#     network_temp.append(NN())
#     network_temp[p].load_state_dict(network.state_dict())
#     network_temp[p].to(device)
