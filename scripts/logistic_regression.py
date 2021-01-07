import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset import combine_data
from dataset import SwissDialectDataset
from trainer import Trainer


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, vec: torch.FloatTensor):
        return torch.sigmoid(self.linear(vec))


class LogisticRegressionTrainer(Trainer):

    def train(
        self,
        model: LogisticRegression,
        train_dataset: SwissDialectDataset,
        dev_dataset: SwissDialectDataset,
    ):

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)

        train_losses = []
        train_counter = []
        test_losses = []
        test_counter = [i * len(train_loader.dataset) for i in range(self.n_epochs + 1)]

        for epoch in range(1, self.n_epochs + 1):
            for batch_id, (ivector_batch, batch_labels) in enumerate(train_loader):
                ivector_batch = Variable(ivector_batch)
                batch_labels = Variable(batch_labels)

                optimizer.zero_grad()
                outputs = model(ivector_batch)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                if batch_id % self.logging_interval == 0:
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
                        (batch_id * self.batch_size)
                        + ((epoch - 1) * len(train_loader.dataset))
                    )

            test_loss = self.test(model, dev_dataset)
            test_losses.append(test_loss)

        return model, train_losses, train_counter, test_losses, test_counter

    def test(self, model: LogisticRegression, dev_dataset: SwissDialectDataset):

        dev_loader = DataLoader(dataset=dev_dataset, batch_size=self.batch_size)
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
    dev_vec_file = "data/dev.vec"
    dev_txt_file = "data/dev.txt"

    all_ivectors, all_labels = combine_data(
        train_vec_file, train_txt_file, dev_vec_file, dev_txt_file
    )

    model_type = LogisticRegression

    trainer = LogisticRegressionTrainer(model_type, all_ivectors, all_labels, n_epochs=3)
    best_model = trainer.cross_validation()
