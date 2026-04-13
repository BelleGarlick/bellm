import os
import shutil

from torch.utils.tensorboard import SummaryWriter


class TensorBoardInterface:

    def __init__(self, log_dir):
        self.live_path = os.path.join(log_dir, 'live')
        self.train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
        self.val_writer = SummaryWriter(os.path.join(log_dir, 'validation'))
        self.live_writer = SummaryWriter(self.live_path)

    def log_current_training_data(self, train_loss, validation_loss, batch_step):
        self.train_writer.add_scalar("loss/batch", train_loss, global_step=batch_step)
        self.train_writer.add_scalar("loss/batch", validation_loss, global_step=batch_step)

    def log_epoch_data(self, train_loss, validation_loss, learning_rate, epoch):
        self.train_writer.add_scalar("loss/epoch", train_loss, global_step=epoch)
        self.val_writer.add_scalar("loss/epoch", validation_loss, global_step=epoch)

        self.train_writer.add_scalar("learning_rate", learning_rate, global_step=epoch)
        self.val_writer.add_scalar("learning_rate", learning_rate, global_step=epoch)

    def log_training_epoch_data(self, epoch, bidx, batch_count, batch_training_loss):
        if bidx == 0 and os.path.exists(self.live_path):
            self.live_writer.close()
            shutil.rmtree(self.live_path)
            self.live_writer = SummaryWriter(self.live_path)

        live_message = f"[Training] Epoch: {epoch} | Batch: {bidx}/{batch_count} | Loss: {batch_training_loss:.4f}"
        print(f"\r{live_message}", end="")
        self.live_writer.add_scalar(
            "live/training-loss",
            batch_training_loss,
            global_step=bidx
        )

    def log_validation_epoch_data(self, epoch, bidx, batch_count, batch_validation_loss):
        live_message = f"[Validating] Epoch: {epoch} | Batch: {bidx}/{batch_count} | Loss: {batch_validation_loss:.4f}"
        print(f"\r{live_message}", end="")
        self.live_writer.add_scalar(
            "live/validation-loss",
            batch_validation_loss,
            global_step=bidx
        )

    def log_test_text(self, title, message, epoch):
        self.val_writer.add_text(
            f"{title}",
            message,
            global_step=epoch
        )
