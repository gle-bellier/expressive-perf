import torch
import matplotlib.pyplot as plt


def logging(log,
            logger,
            mode,
            idx,
            reg,
            c_dict,
            loss_dict,
            inv_transform,
            ddsp=None):

    # plot regularization

    if reg:
        log(f"{mode}/f0_loss", loss_dict["f0"])
        log(f"{mode}/lo_loss", loss_dict["lo"])

    # plot adversarial training
    logger.experiment.add_scalars(
        f"{mode}/adv",
        {
            'gen': loss_dict["G"],
            'disc': loss_dict["D"]
        },
        #global_idx=idx,
    )

    if idx % 10 == 0:

        u_f0, u_lo = post_processing(c_dict["u"], inv_transform)
        e_f0, e_lo = post_processing(c_dict["e"], inv_transform)
        g_f0, g_lo = post_processing(c_dict["g"], inv_transform)

        if reg:
            plt.plot(u_f0[0].squeeze().cpu().detach(), label="u_f0")
        plt.plot(g_f0[0].squeeze().cpu().detach(), label="g_f0")
        plt.legend()
        logger.experiment.add_figure("contours/gen/f0", plt.gcf(), idx)

        plt.plot(e_f0[0].squeeze().cpu().detach(), label="e_f0")
        plt.legend()
        logger.experiment.add_figure("contours/sample/f0", plt.gcf(), idx)

        if reg:
            plt.plot(u_lo[0].squeeze().cpu().detach(), label="u_lo")
        plt.plot(g_lo[0].squeeze().cpu().detach(), label="g_lo")
        plt.legend()
        logger.experiment.add_figure("contours/gen/lo", plt.gcf(), idx)

        plt.plot(e_lo[0].squeeze().cpu().detach(), label="e_lo")
        plt.legend()
        logger.experiment.add_figure("contours/sample/lo", plt.gcf(), idx)

    # listen to audio
    if idx % 10 == 0:

        if ddsp is not None:

            wav = c2wav(c_dict["g"][0:1], inv_transform, ddsp).detach()
            wav = wav.reshape(-1).cpu().numpy()
            logger.experiment.add_audio(
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


def c2wav(c, inv_transform, ddsp):
    f0, lo = post_processing(c, inv_transform)
    f0 = f0.permute(0, 2, 1)
    lo = lo.permute(0, 2, 1)

    wav = ddsp(f0, lo)

    # B L C -> B C L
    wav = wav.permute(0, 2, 1)

    return wav
