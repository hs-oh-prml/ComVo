import torch

from exp.discriminators import cMultiResolutionDiscriminator
from exp.feature_extractors import FeatureExtractor
from exp.heads import FourierHead
from exp.helpers import plot_spectrogram_to_numpy
from exp.models import Backbone
from exp.modules import safe_log
from exp.experiment import ComVoExp
from exp.loss import cFeatureMatchingLoss, cGeneratorLoss, cDiscriminatorLoss


class ComVoExp(ComVoExp):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
        sample_rate: int,
        initial_learning_rate: float,
        num_warmup_steps: int = 0,
        mel_loss_coeff: float = 45,
        mrd_loss_coeff: float = 1.0,
        pretrain_mel_steps: int = 0,
        decay_mel_coeff: bool = False,
        evaluate_utmos: bool = False,
        evaluate_pesq: bool = False,
        evaluate_periodicty: bool = False,
    ):
        super().__init__(
            feature_extractor,
            backbone,
            head,
            sample_rate,
            initial_learning_rate,
            num_warmup_steps,
            mel_loss_coeff,
            mrd_loss_coeff,
            pretrain_mel_steps,
            decay_mel_coeff,
            evaluate_utmos,
            evaluate_pesq,
            evaluate_periodicty,
        )

        self.multiresddisc = cMultiResolutionDiscriminator()
        self.cgen_loss = cGeneratorLoss()
        self.cdisc_loss = cDiscriminatorLoss()
        self.cfeat_matching_loss = cFeatureMatchingLoss()

    def training_step(self, batch, batch_idx, optimizer_idx, **kwargs):
        audio_input = batch

        if optimizer_idx == 0 and self.train_discriminator:
            with torch.no_grad():
                audio_hat = self(audio_input, **kwargs)

            real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(
                y=audio_input,
                y_hat=audio_hat,
                **kwargs,
            )
            real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(
                y=audio_input,
                y_hat=audio_hat,
                **kwargs,
            )
            loss_mp, loss_mp_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp
            )
            loss_mrd, loss_mrd_real, _ = self.cdisc_loss(
                disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
            )
            loss_mp /= len(loss_mp_real)
            loss_mrd /= len(loss_mrd_real)
            loss = loss_mp + self.hparams.mrd_loss_coeff * loss_mrd

            self.log("discriminator/total", loss, prog_bar=True)
            self.log("discriminator/multi_period_loss", loss_mp)
            self.log("discriminator/multi_res_loss", loss_mrd)
            return loss

        # train generator
        if optimizer_idx == 1:
            audio_hat = self(audio_input, **kwargs)
            if self.train_discriminator:
                _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(
                    y=audio_input,
                    y_hat=audio_hat,
                    **kwargs,
                )
                _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(
                    y=audio_input,
                    y_hat=audio_hat,
                    **kwargs,
                )
                loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
                loss_gen_mrd, list_loss_gen_mrd = self.cgen_loss(
                    disc_outputs=gen_score_mrd
                )
                loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
                loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
                loss_fm_mp = self.feat_matching_loss(
                    fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp
                ) / len(fmap_rs_mp)
                loss_fm_mrd = self.cfeat_matching_loss(
                    fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd
                ) / len(fmap_rs_mrd)

                self.log("generator/multi_period_loss", loss_gen_mp)
                self.log("generator/multi_res_loss", loss_gen_mrd)
                self.log("generator/feature_matching_mp", loss_fm_mp)
                self.log("generator/feature_matching_mrd", loss_fm_mrd)
            else:
                loss_gen_mp = loss_gen_mrd = loss_fm_mp = loss_fm_mrd = 0

            mel_loss = self.melspec_loss(audio_hat, audio_input)
            loss = (
                loss_gen_mp
                + self.hparams.mrd_loss_coeff * loss_gen_mrd
                + loss_fm_mp
                + self.hparams.mrd_loss_coeff * loss_fm_mrd
                + self.mel_loss_coeff * mel_loss
            )

            self.log("generator/total_loss", loss, prog_bar=True)
            self.log("mel_loss_coeff", self.mel_loss_coeff)
            self.log("generator/mel_loss", mel_loss)

            if self.global_step % 1000 == 0 and self.global_rank == 0:
                self.logger.experiment.add_audio(
                    "train/audio_in",
                    audio_input[0].data.cpu(),
                    self.global_step,
                    self.hparams.sample_rate,
                )
                self.logger.experiment.add_audio(
                    "train/audio_pred",
                    audio_hat[0].data.cpu(),
                    self.global_step,
                    self.hparams.sample_rate,
                )
                with torch.no_grad():
                    mel = safe_log(self.melspec_loss.mel_spec(audio_input[0]))
                    mel_hat = safe_log(self.melspec_loss.mel_spec(audio_hat[0]))
                self.logger.experiment.add_image(
                    "train/mel_target",
                    plot_spectrogram_to_numpy(mel.data.cpu().numpy()),
                    self.global_step,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    "train/mel_pred",
                    plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
                    self.global_step,
                    dataformats="HWC",
                )

            return loss
