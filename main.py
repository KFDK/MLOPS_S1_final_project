import argparse
import sys
import pdb
import torch

from data import MNIST_corrupted
from model import MyAwesomeModel
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = fc_model.Network(
        checkpoint["input_size"], checkpoint["output_size"], checkpoint["hidden_layers"]
    )
    model.load_state_dict(checkpoint["state_dict"])

    return model


class TrainOREvaluate(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>",
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")

            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # TODO: Implement training loop here
        model = MyAwesomeModel()
        data_path = "../../../data/corruptmnist/"
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = MNIST_corrupted(
            data_path=data_path, data_type="train", transform=transform
        )
        trainloader = DataLoader(train_dataset, batch_size=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        steps = 0

        epochs = 2
        criterion = nn.NLLLoss()

        train_losses = []

        for e in range(epochs):
            model.train()
            running_loss = 0
            for images, labels in trainloader:
                steps += 1
                images.resize_(images.size()[0], 784)

                optimizer.zero_grad()

                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            else:
                train_losses.append(running_loss / len(trainloader))
                print(f"Training loss: {running_loss/len(trainloader)}")

        # save model
        torch.save(model.state_dict(), "trained_model.pth")

        # plotting
        plt.plot(np.arange(0, epochs), train_losses)
        plt.savefig("train_loss_vs_epoch.png")
        img = Image.open("train_loss_vs_epoch.png")
        img.show()

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("load_model_from", default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        model.load_state_dict(torch.load(args.load_model_from))
        # model = torch.load(args.load_model_from)
        data_path = "../../../data/corruptmnist/"
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = MNIST_corrupted(
            data_path=data_path, data_type="test", transform=transform
        )
        testloader = DataLoader(test_dataset, batch_size=64)

        with torch.no_grad():
            model.eval()
            running_accs = []
            for images, labels in testloader:
                log_ps = model(images)

                top_p, top_class = log_ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                running_accs.append(accuracy.item())
        accuracy = sum(running_accs) / len(running_accs)
        print(f"Accuracy: {accuracy*100}%")


if __name__ == "__main__":
    TrainOREvaluate()
