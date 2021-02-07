import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import sys
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from dataset import load_ivectors
from dataset import load_labels
from dataset import SwissDialectDataset


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(4, 10)  # The 4 is for the four classes.

        # TODO: Maybe remove one of the linear layers (the one with 128 first).
        self.model = nn.Sequential(
            nn.Linear(410, 512),  # 410 is num features + embeddings
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),  # The 1 is the number converted to a probability.
            nn.Sigmoid(),
        )

    def forward(self, x, labels):
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(4, 10)  # The 4 is for the four classes.

        self.model = nn.Sequential(
            nn.Linear(110, 128),  # 110 is 100 random datapoints + embeddings
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 400),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.squeeze()


def generator_train_step(
    batch_size, discriminator, generator, gen_optimizer, criterion
):
    gen_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 100))
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 4, batch_size)))
    fake_ivectors = generator(z, fake_labels)
    validity = discriminator(fake_ivectors, fake_labels)
    gen_loss = criterion(validity, Variable(torch.ones(batch_size)))
    gen_loss.backward()
    gen_optimizer.step()
    return gen_loss.item()


def discriminator_train_step(
    batch_size,
    discriminator,
    generator,
    discr_optimizer,
    criterion,
    real_ivectors,
    labels,
):
    discr_optimizer.zero_grad()

    # Train with real ivectors.
    real_validity = discriminator(real_ivectors, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)))

    # Train with fake ivectors.
    z = Variable(torch.randn(batch_size, 100))
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 4, batch_size)))
    fake_ivectors = generator(z, fake_labels)
    fake_validity = discriminator(fake_ivectors, fake_labels)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)))

    discr_loss = real_loss + fake_loss
    discr_loss.backward()
    discr_optimizer.step()
    return discr_loss.item()


def generate_ivector(generator, dialect):
    z = Variable(torch.randn(1, 100))
    label = torch.LongTensor([dialect])
    ivector = generator(z, label).data
    return ivector


def store_generated_data(generator, num_samples):
    dialects = [[x] * num_samples for x in (0, 1, 2, 3)]
    dialects = [item for sublist in dialects for item in sublist]

    with open(
        "data/gan-ivectors-" + str(num_samples) + ".csv", "w"
    ) as gen_ivec_file, open(
        "data/gan-labels-" + str(num_samples) + ".txt", "w"
    ) as gen_label_file:

        ivec_writer = csv.writer(gen_ivec_file)
        label_writer = csv.writer(gen_label_file, delimiter="\t")

        for dialect in dialects:
            gen_ivecs = generate_ivector(generator, dialect)
            ivec_writer.writerow(gen_ivecs.tolist())

            txt_file_out = ["GAN-generated artificial text", dialect]
            label_writer.writerow(txt_file_out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Train and config must always be used together.
    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Train a GAN that can generate artificial ivectors.",
    )
    # parser.add_argument("-c", "--config_file", help="A configuration file that specifies training parameters.")

    # A model can be trained and then directly used for generation.
    # Or a model stored in a file can be used (flag: --load_model).
    parser.add_argument(
        "-g",
        "--generate",
        type=int,
        help="Generate new samples and specify the number of samples you want.",
    )
    parser.add_argument("-l", "--load_model", type=str, help="Use an existing model for generation.")

    args = parser.parse_args()

    # Training process.
    if args.train:
        TRAIN_VEC_FILE = "data/train.vec"
        TRAIN_TXT_FILE = "data/train.txt"

        batch_size = 32
        num_epochs = 100
        lr = 1e-4

        train_ivectors = load_ivectors(TRAIN_VEC_FILE)
        train_labels = load_labels(TRAIN_TXT_FILE)
        dataset = SwissDialectDataset(train_ivectors, train_labels)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        writer = SummaryWriter()

        generator = Generator()
        discriminator = Discriminator()

        criterion = nn.BCELoss()
        discr_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

        for epoch in range(num_epochs):
            print("Epoch {}...".format(epoch + 1), end=" ")
            for i, (ivectors, labels) in enumerate(data_loader):

                step = epoch * len(data_loader) + i + 1
                real_ivectors = Variable(ivectors)
                labels = Variable(labels)
                generator.train()

                d_loss = discriminator_train_step(
                    len(real_ivectors),
                    discriminator,
                    generator,
                    discr_optimizer,
                    criterion,
                    real_ivectors,
                    labels,
                )

                g_loss = generator_train_step(
                    batch_size, discriminator, generator, gen_optimizer, criterion
                )

                writer.add_scalars("scalars", {"g_loss": g_loss, "d_loss": d_loss}, step)

            print("Done!")

        store = input("\n\nDo you want to store the GAN generator? [yes|no]: ")
        if store == "yes":
            model_file_name = input("\n\nPlease choose a file name for the model: gan-")
            torch.save(generator.state_dict(), "stored_models/gan-" + model_file_name + ".pt")
        else:
            print("Generator model is not stored.")

    # Generation Process.
    if args.generate:
        if args.load_model:
            generator = Generator()
            generator.load_state_dict(torch.load(args.load_model))
            generator.eval()
        if not args.load_model and not args.train:
            print("You must either train a model or use a stored model for generation.")
            sys.exit()

        num_samples = args.generate
        store_generated_data(generator, num_samples)
