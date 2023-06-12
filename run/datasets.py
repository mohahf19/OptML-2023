from indexed_dataset import IndexedDataset
from torchvision import datasets, transforms

# returns normalized and tranformed CIFAR10 dataset
def get_cifar10_data():
    transform_train = transforms.Compose([
         transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    
    train_dataset = datasets.CIFAR10(
            root="data", train=True,
            download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform_test
    )

    return IndexedDataset(train_dataset), IndexedDataset(test_dataset)

# returns normalized and tranformed MNIST dataset
def get_mnist_data():
    transform = transforms.Compose(
          [transforms.Resize((32, 32)),
          transforms.ToTensor(),
          transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        "data", train=False, download=True, transform=transform
    )

    return IndexedDataset(train_dataset), IndexedDataset(test_dataset)

