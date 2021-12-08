import torch


class LSGAN_loss:
    def __init__(self):
        """Loss used for LSGANs
        """
        pass

    def disc_loss(self, dx: torch.Tensor, dgz: torch.Tensor) -> torch.Tensor:
        """Compute the discriminator loss in the LSGAN fashion

        Args:
            dx (torch.Tensor): output of the discriminator for real samples
            dgz (torch.Tensor): output of the discriminator for generated samples

        Returns:
            torch.Tensor: loss of the discriminator 
        """

        loss_fake = 0.5 * torch.mean((dx - 1)**2)
        loss_real = 0.5 * torch.mean(dgz**2)

        return loss_fake + loss_real

    def gen_loss(self, dx, dgz):
        """Compute the generator loss in the LSGAN fashion

        Args:
            dx (torch.Tensor): output of the discriminator for real samples
            dgz (torch.Tensor): output of the discriminator for generated samples

        Returns:
            torch.Tensor: loss of the generator 
        """

        loss_gen = 0.5 * torch.mean((dgz - 1)**2)
        return loss_gen
