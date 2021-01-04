import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset import SwissDialectDataset


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, vec):
        return torch.sigmoid(self.linear(vec))


def train(
    model,
    train_dataset,
    dev_dataset,
    n_epochs=10,
    batch_size=10,
    shuffle=False,
    lr=0.01,
    logging_interval=100,
):

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    for epoch in range(1, n_epochs + 1):
        for batch_id, (ivector_batch, batch_labels) in enumerate(train_loader):
            ivector_batch = Variable(ivector_batch)
            batch_labels = Variable(batch_labels)

            optimizer.zero_grad()
            outputs = model(ivector_batch)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            if batch_id % logging_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_id * len(ivector_batch),
                        len(train_loader.dataset),
                        100.0 * batch_id / len(train_loader),
                        loss.item(),
                    )
                )
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_id * batch_size) + ((epoch - 1) * len(train_loader.dataset))
                )

        test_loss = test(model, dev_dataset)
        test_losses.append(test_loss)

    return model, train_losses, train_counter, test_losses, test_counter


def test(model, dev_dataset, batch_size_test=10, shuffle=False):

    dev_loader = DataLoader(
        dataset=dev_dataset, batch_size=batch_size_test, shuffle=shuffle
    )
    criterion = torch.nn.CrossEntropyLoss()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for ivector_batch, batch_labels in dev_loader:
            output = model(ivector_batch)
            test_loss += criterion(output, batch_labels).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(batch_labels.data.view_as(pred)).sum()

    test_loss /= len(dev_loader.dataset)
    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(dev_loader.dataset),
            100.0 * correct / len(dev_loader.dataset),
        )
    )

    return test_loss


if __name__ == "__main__":

    train_vec_file = "data/train.vec"
    train_txt_file = "data/train.txt"
    train_dataset = SwissDialectDataset(train_vec_file, train_txt_file)

    dev_vec_file = "data/dev.vec"
    dev_txt_file = "data/dev.txt"
    dev_dataset = SwissDialectDataset(dev_vec_file, dev_txt_file)

    input_dim = train_dataset.n_features
    output_dim = train_dataset.n_classes

    model = LogisticRegression(input_dim, output_dim)

    # Test the performance on the dev set with randomly initialized weights.
    # This way it can be compared to the performance after training.
    test(model, dev_dataset)

    model, train_losses, train_counter, test_losses, test_counter = train(
        model, train_dataset, dev_dataset, n_epochs=10
    )
