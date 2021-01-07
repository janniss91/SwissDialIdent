import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset import SwissDialectDataset
from metrics import Metrics
from trainer import Trainer


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, tensor: torch.FloatTensor):
        return torch.sigmoid(self.linear(tensor))


class LogisticRegressionTrainer(Trainer):

    def train(
        self,
        model: LogisticRegression,
        train_dataset: SwissDialectDataset,
        dev_dataset: SwissDialectDataset,
        verbose: bool
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
                    train_losses.append(loss.item())
                    train_counter.append(
                        (batch_id * self.batch_size)    
                        + ((epoch - 1) * len(train_loader.dataset))
                    )

                    if verbose:
                        self.print_train_metrics(epoch, batch_id, ivector_batch, train_loader, loss)

            metrics, test_loss = self.test(model, dev_dataset, verbose)
            test_losses.append(test_loss)

        metrics.store_losses(train_losses, train_counter, test_losses, test_counter)

        return model, metrics

    def test(self, model: LogisticRegression, dev_dataset: SwissDialectDataset, verbose: bool):

        dev_loader = DataLoader(dataset=dev_dataset, batch_size=self.batch_size)
        criterion = torch.nn.CrossEntropyLoss()

        test_loss = 0
        correct = 0
        all_preds = torch.empty(0)
        with torch.no_grad():
            for ivector_batch, batch_labels in dev_loader:
                output = model(ivector_batch)
                test_loss += criterion(output, batch_labels).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(batch_labels.data.view_as(pred)).sum()
                all_preds = torch.cat((all_preds, torch.flatten(pred)))

        # The metrics object requires numpy arrays instead of torch tensors.
        metrics = Metrics(dev_dataset.labels.numpy(), all_preds.numpy())

        test_loss /= len(dev_loader.dataset)

        if verbose:
            self.print_test_metrics(test_loss, correct, dev_loader, metrics)

        return metrics, test_loss
