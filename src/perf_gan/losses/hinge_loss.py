import torch


class Hinge_loss:
    def __init__(self):
        """Hinge loss for GANs
        """
        pass

    def disc_loss(self, dx: torch.Tensor, dgz: torch.Tensor) -> torch.Tensor:
        """Compute the discriminator term in the Hinge loss

        Args:
            dx (torch.Tensor): output of the discriminator for real samples
            dgz (torch.Tensor): output of the discriminator for generated samples

        Returns:
            torch.Tensor: loss of the discriminator 
        """

        zeros = torch.zeros_like(dx)
        loss_real = -torch.mean(torch.min(zeros, -1 + dx))
        loss_fake = -torch.mean(torch.min(zeros, -1 - dgz))

        return loss_real + loss_fake

    def gen_loss(self, dx: torch.Tensor, dgz: torch.Tensor) -> torch.Tensor:
        """Compute the generator term in the Hinge loss

        Args:
            dx (torch.Tensor): output of the generator for real samples
            dgz (torch.Tensor): output of the generator for generated samples

        Returns:
            torch.Tensor: loss of the generator 
        """

        loss_real = 0  #torch.mean(dx)
        loss_fake = -torch.mean(dgz)

        return loss_real + loss_fake
