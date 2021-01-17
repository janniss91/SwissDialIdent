import time
import torch
from numpy import ndarray
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
        model_type: LogisticRegression,
        train_ivecs: ndarray,
        train_labels: ndarray, 
        test_ivecs: ndarray,
        test_labels: ndarray,
        verbose: bool,
    ):
        # Set up PyTorch compatible datasets and dataloader.
        train_dataset = SwissDialectDataset(train_ivecs, train_labels)
        test_dataset = SwissDialectDataset(test_ivecs, test_labels)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size)

        # Initialize and prepare model for training.
        input_dim = train_dataset.n_features
        output_dim = train_dataset.n_classes
        model = self.model_type(input_dim, output_dim)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)

        self.logger.train_samples = train_dataset.n_samples
        self.logger.test_samples = test_dataset.n_samples
        
        train_losses = []
        train_counter = []
        test_losses = []
        test_counter = [i * len(train_loader.dataset) for i in range(self.n_epochs + 1)]

        for epoch in range(1, self.n_epochs + 1):
            # Track date, time and training time.
            train_time = time.strftime("%a-%d-%b-%Y-%H:%M:%S", time.localtime())
            start_time = time.time()
            for batch_id, (ivector_batch, batch_labels) in enumerate(train_loader):
                ivector_batch = Variable(ivector_batch)
                batch_labels = Variable(batch_labels)
                optimizer.zero_grad()
                outputs = model(ivector_batch)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                # Store training losses.
                if batch_id % self.log_interval == 0:
                    train_losses.append(loss.item())
                    train_counter.append((epoch, (batch_id * self.batch_size)))

                    # Print training losses.
                    if verbose:
                        self.print_train_metrics(
                            epoch, batch_id, len(train_loader.dataset), loss
                        )

            # Test and store test losses.
            metrics, test_loss = self.test(model, test_dataset, verbose)
            test_losses.append(test_loss)

            # Set up logging parameters and write metrics to logs.
            self.logger.epoch_no = epoch
            end_time = time.time()
            runtime = round(end_time - start_time, 2)
            self.logger.log_metrics(train_time, runtime, metrics)

        # Store losses to metrics and write losses to logs.
        metrics.store_losses(train_losses, train_counter, test_losses, test_counter)
        self.logger.log_losses(train_time, metrics)

        return model, metrics

    def test(
        self, model: LogisticRegression, test_dataset: SwissDialectDataset, verbose: bool
    ):
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size)
        criterion = torch.nn.CrossEntropyLoss()

        test_loss = 0
        correct = 0
        all_preds = torch.empty(0)
        with torch.no_grad():
            for ivector_batch, batch_labels in test_loader:
                output = model(ivector_batch)
                test_loss += criterion(output, batch_labels).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(batch_labels.data.view_as(pred)).sum()
                all_preds = torch.cat((all_preds, torch.flatten(pred)))

        # The metrics object requires numpy arrays instead of torch tensors.
        metrics = Metrics(test_dataset.labels.numpy(), all_preds.numpy())

        test_loss /= len(test_loader.dataset)

        if verbose:
            self.print_test_metrics(test_loss, correct, test_dataset.n_samples, metrics)

        return metrics, test_loss
