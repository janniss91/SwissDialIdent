import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from dataset import load_ivectors
from dataset import load_labels
from dataset import SwissDialectDataset


TRAIN_VEC_FILE = "data/train.vec"
TRAIN_TXT_FILE = "data/train.txt"

train_ivectors = load_ivectors(TRAIN_VEC_FILE)
train_labels = load_labels(TRAIN_TXT_FILE)

dataset = SwissDialectDataset(train_ivectors, train_labels)

batch_size = 32
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(4, 10)  # The 4 is for the four classes.

        # TODO: Maybe remove one of the linear layers (the one with 128 first).
        self.model = nn.Sequential(
            nn.Linear(404, 512),  # 404 is num features + embeddings
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
        # TODO: The view can probably be omitted (because it does nothing).
        x = x.view(x.size(0), 400)  ## 400 is the number of features in i-vectors.
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(4, 10)  # The 4 is for the four classes.

        self.model = nn.Sequential(
            nn.Linear(104, 128),  # 104 is 100 random datapoints + embeddings
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 400),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        # TODO: The view can probably be omitted (because it does nothing).
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out


def generator_train_step(batch_size, discriminator, generator, gen_optimizer, criterion):
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 100))
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 4, batch_size)))
    fake_ivectors = generator(z, fake_labels)
    validity = discriminator(fake_ivectors, fake_labels)
    gen_loss = criterion(validity, Variable(torch.ones(batch_size)))
    gen_loss.backward()
    gen_optimizer.step()
    return g_loss.item()


def discriminator_train_step(
    batch_size, discriminator, generator, discr_optimizer, criterion, real_ivectors, labels
):
    d_optimizer.zero_grad()

    # Train with real ivectors.
    real_validity = discriminator(real_ivectors, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)))

    # Train with fake ivectors.
    z = Variable(torch.randn(batch_size, 100))
    fake_labels = Variable(
        torch.LongTensor(np.random.randint(0, 4, batch_size))
    )
    fake_ivectors = generator(z, fake_labels)
    fake_validity = discriminator(fake_ivectors, fake_labels)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)))

    discr_loss = real_loss + fake_loss
    discr_loss.backward()
    discr_optimizer.step()
    return d_loss.item()


writer = SummaryWriter()

generator = Generator()
discriminator = Discriminator()

criterion = nn.BCELoss()
discr_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)

num_epochs = 100
n_critic = 5  # TODO: variable is not used?
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

    torch.save(generator.state_dict(), 'generator_state.pt')


def generate_ivector(generator, dialect):
    z = Variable(torch.randn(1, 100))
    label = torch.LongTensor([dialect])
    ivector = generator(z, label).data
    return ivector


# Model loading process:

# model = Generator(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()
# generate_digit(model, dialect)
