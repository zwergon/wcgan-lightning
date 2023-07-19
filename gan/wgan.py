import torch
import torchvision
import lightning as L
import torch.nn.functional as F
import torch.nn as nn

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from gan.generator import Generator
from gan.critic import Critic

def get_image_grid(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    return make_grid(image_unflat[:num_images], nrow=5)


class WGAN(L.LightningModule):
    def __init__(
        self,
        channels,
        width,
        height, 
        z_dim: int = 64,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        c_lambda: float = 10,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # networks
        self.generator = Generator(z_dim=z_dim, im_chan=channels)
        self.critic = Critic()
      
        self.generator = self.generator.apply(self.weights_init)
        self.critic = self.critic.apply(self.weights_init)

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
    

    def get_gradient(self, real, fake):
        '''
        Return the gradient of the critic's scores with respect to mixes of real and fake images.
        Parameters:
            crit: the critic model
            real: a batch of real images
            fake: a batch of fake images
            
        Returns:
            gradient: the gradient of the critic's scores, with respect to the mixed image
        '''

        #epsilon: a vector of the uniformly random proportions of real/fake per mixed image
        epsilon = torch.rand(len(real), 1, 1, 1, device=real.device, requires_grad=True)

        # Mix the images together
        mixed_images = real * epsilon + fake * (1 - epsilon)

        # Calculate the critic's scores on the mixed images
        mixed_scores = self.critic(mixed_images)
        
        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            # Note: You need to take the gradient of outputs with respect to inputs.
            # This documentation may be useful, but it should not be necessary:
            # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
            inputs=mixed_images,
            outputs=mixed_scores,
            # These other parameters have to do with the pytorch autograd engine works
            grad_outputs=torch.ones_like(mixed_scores), 
            create_graph=True,
            retain_graph=True,
        )[0]
        return gradient
    
    @staticmethod
    def gradient_penalty(gradient):
        '''
        Return the gradient penalty, given a gradient.
        Given a batch of image gradients, you calculate the magnitude of each image's gradient
        and penalize the mean quadratic distance of each magnitude to 1.
        Parameters:
            gradient: the gradient of the critic's scores, with respect to the mixed image
        Returns:
            penalty: the gradient penalty
        '''
        # Flatten the gradients so that each row captures one image
        gradient = gradient.view(len(gradient), -1)

        # Calculate the magnitude of every row
        gradient_norm = gradient.norm(2, dim=1)
        
        # Penalize the mean squared distance of the gradient norms from 1
        penalty = torch.mean((gradient_norm - 1)**2)

        return penalty
    
    def critic_criterion(self, crit_fake_pred, crit_real_pred, gp):
        '''
        Return the loss of a critic given the critic's scores for fake and real images,
        the gradient penalty, and gradient penalty weight.
        Parameters:
            crit_fake_pred: the critic's scores of the fake images
            crit_real_pred: the critic's scores of the real images
            gp: the unweighted gradient penalty
            
        Returns:
            a scalar for the critic's loss, accounting for the relevant factors
        '''

        #c_lambda: the current weight of the gradient penalty 
        c_lambda = self.hparams.c_lambda
        
        loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp
        
        return loss
    
    def gen_criterion(self, crit_fake_pred):
        '''
        Return the loss of a generator given the critic's scores of the generator's fake images.
        Parameters:
            crit_fake_pred: the critic's scores of the fake images
        Returns:
            a scalar loss value for the current batch of the generator
        '''
        loss = -1. * torch.mean(crit_fake_pred)
        return loss

 
    def get_critic_loss(self, batch, batch_idx):

        real, _ = batch

         # sample noise
        fake_noise = Generator.get_noise(real.shape[0], self.hparams.z_dim, device=real.device)
        
        # generate images
        fake = self.generator(fake_noise)

        crit_fake_pred = self.critic(fake.detach())
        crit_real_pred = self.critic(real)
       
        gradient = self.get_gradient(real, fake.detach())
        gp = WGAN.gradient_penalty(gradient)
        
        return self.critic_criterion(crit_fake_pred, crit_real_pred, gp)
        
       
    def get_gen_loss(self, batch):

        imgs, _ = batch

        fake_noise = Generator.get_noise(imgs.shape[0], self.hparams.z_dim, device=imgs.device)
      
        fake = self.generator(fake_noise)
        crit_fake_pred = self.critic(fake)

        return self.gen_criterion(crit_fake_pred)
    

    def training_step(self, batch, batch_idx):
      
        optimizer_g, optimizer_c = self.optimizers()

        # Optimize discriminator
        # Measure discriminator's ability to classify real from generated samples
        
        self.toggle_optimizer(optimizer_c)
        
        critic_loss = self.get_critic_loss(batch, batch_idx=batch_idx)
        self.log("crit_loss", critic_loss, prog_bar=True)
        optimizer_c.zero_grad()
        self.manual_backward(critic_loss, retain_graph=True)
        optimizer_c.step()
        self.untoggle_optimizer(optimizer_c)

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
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_c], []

    