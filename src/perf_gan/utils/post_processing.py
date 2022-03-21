import torch
import matplotlib.pyplot as plt


def logging(model, mode, c_dict, loss_dict):

    # plot regularization

    if model.reg:
        model.log(f"{mode}/f0_loss", loss_dict["f0"])
        model.log(f"{mode}/lo_loss", loss_dict["lo"])

    idx = model.train_idx if mode == "train" else model.val_idx

    # plot adversarial training
    model.logger.experiment.add_scalars(
        f"{mode}/adv",
        {
            'gen': loss_dict["G"],
            'disc': loss_dict["D"]
        },
        idx,
    )

    if idx % 10 == 0:

        u_f0, u_lo = post_processing(c_dict["u"], model.inv_transform)
        e_f0, e_lo = post_processing(c_dict["e"], model.inv_transform)
        g_f0, g_lo = post_processing(c_dict["g"], model.inv_transform)

        if model.reg:
            plt.plot(u_f0[0].squeeze().cpu().detach(), label="u_f0")
        plt.plot(g_f0[0].squeeze().cpu().detach(), label="g_f0")
        plt.legend()

        model.logger.experiment.add_figure(f"{mode}/gen/f0", plt.gcf(), idx)

        plt.plot(e_f0[0].squeeze().cpu().detach(), label="e_f0")
        plt.legend()
        model.logger.experiment.add_figure(f"{mode}/sample/f0", plt.gcf(), idx)

        if model.reg:
            plt.plot(u_lo[0].squeeze().cpu().detach(), label="u_lo")
        plt.plot(g_lo[0].squeeze().cpu().detach(), label="g_lo")
        plt.ylim([-10, 0])
        plt.legend()
        model.logger.experiment.add_figure(f"{mode}/gen/lo", plt.gcf(), idx)

        plt.plot(e_lo[0].squeeze().cpu().detach(), label="e_lo")
        plt.legend()
        plt.ylim([-10, 0])
        model.logger.experiment.add_figure(f"{mode}/sample/lo", plt.gcf(), idx)

    # listen to audio
    if idx % 10 == 0:

        if model.ddsp is not None:

            wav = c2wav(model, c_dict["g"][0:1]).detach()
            wav = wav.reshape(-1).cpu().numpy()
            model.logger.experiment.add_audio(
                "generation",
                wav,
                idx,
                16000,
            )


def midi2hz(x):
    return torch.pow(2, (x - 69) / 12) * 440


def post_processing(c, inv_transform):
    f0, lo = c.split(1, -2)
    f0, lo = inv_transform(f0, lo)

    # convert midi to hz
    f0 = midi2hz(f0)

    return f0, lo


def c2wav(model, c):
    f0, lo = post_processing(c, model.inv_transform)
    f0 = f0.permute(0, 2, 1)
    lo = lo.permute(0, 2, 1)

    wav = model.ddsp(f0, lo)

    # B L C -> B C L
    wav = wav.permute(0, 2, 1)

    return wav
