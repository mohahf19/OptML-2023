# print("Training with SVRG")
#
# # training parameters
# batch_size = 128
# batch_full_grads = 2**20
#
# device_c = args.device
# train_loader_c = torch.utils.data.DataLoader(
#     train_dataset, batch_size=batch_size_c, shuffle=True
# )
# test_loader_c = torch.utils.data.DataLoader(
#     test_dataset, batch_size=batch_full_grads, shuffle=True
# )
#
# train_loader_temp_c = torch.utils.data.DataLoader(
#     train_dataset, batch_size=batch_full_grads, shuffle=True
# )
#
# network = NN()
# network_temp = NN()
# network_temp.load_state_dict(network.state_dict())
# network.to(device)
# network_temp.to(device)
