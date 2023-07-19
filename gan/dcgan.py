import torch
import torchvision
import lightning as L
import torch.nn.functional as F
import torch.nn as nn

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from gan.generator import Generator
from gan.discriminator import Discriminator

def get_image_grid(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    return make_grid(image_unflat[:num_images], nrow=5)


class DCGAN(L.LightningModule):
    def __init__(
        self,
        channels,
        width,
        height, 
        z_dim: int = 64,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # networks
        self.generator = Generator(z_dim=z_dim, im_chan=channels)
        self.discriminator = Discriminator()
        self.criterion = nn.BCEWithLogitsLoss()

        self.generator = self.generator.apply(self.weights_init)
        self.discriminator = self.discriminator.apply(self.weights_init)

    # You initialize the weights to the normal distribution
    # with mean 0 and standard deviation 0.02
    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
       

    def forward(self, z):
        return self.generator(z)
 
    def get_disc_loss(self, batch, batch_idx):

        imgs, _ = batch

         # sample noise
        fake_noise = Generator.get_noise(imgs.shape[0], self.hparams.z_dim, device=imgs.device)
        
        # generate images
        fake = self.generator(fake_noise)

        disc_fake_pred = self.discriminator(fake.detach())
        disc_fake_loss = self.criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        
        
        disc_real_pred = self.discriminator(imgs)
        disc_real_loss = self.criterion(disc_real_pred,  torch.ones_like(disc_real_pred))

        if self.global_step % 500 == 0:
            fake_grid = get_image_grid(fake)
            self.logger.experiment.add_image("real_images", fake_grid, self.global_step)

        return (disc_fake_loss + disc_real_loss) / 2
        

    def get_gen_loss(self, batch):

        imgs, _ = batch

        fake_noise = Generator.get_noise(imgs.shape[0], self.hparams.z_dim, device=imgs.device)
      
        fake = self.generator(fake_noise)
        disc_fake_pred = self.discriminator(fake)

        return self.criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    

    def training_step(self, batch, batch_idx):
      
        optimizer_g, optimizer_d = self.optimizers()

        # Optimize discriminator
        # Measure discriminator's ability to classify real from generated samples
        
        self.toggle_optimizer(optimizer_d)
        
        disc_loss = self.get_disc_loss(batch, batch_idx=batch_idx)
        self.log("disc_loss", disc_loss, prog_bar=True)
        optimizer_d.zero_grad()
        self.manual_backward(disc_loss, retain_graph=True)
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)

        # train generator
        self.toggle_optimizer(optimizer_g)
        gen_loss = self.get_gen_loss(batch)
        self.log("gen_loss", gen_loss, prog_bar=True)
    
        optimizer_g.zero_grad()
        self.manual_backward(gen_loss)
        optimizer_g.step()
        
        self.untoggle_optimizer(optimizer_g)

       
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    