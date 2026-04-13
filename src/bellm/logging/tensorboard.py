from typing import List, Tuple

import mlflow


class MLflowInterface:

    def __init__(self, experiment_name, run_name=None):
        """
        Initializes the MLflow experiment and starts a run.
        """
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(experiment_name)
        # This starts a global run. In a real training script,
        # you might want to manage this with 'with mlflow.start_run():'
        # outside this class, but for a direct replacement:
        if mlflow.active_run() is None:
            mlflow.start_run(run_name=run_name)

    def log_epoch_data(self, train_loss, validation_loss, learning_rate, epoch):
        metrics = {
            "loss/train": train_loss,
            "loss/val": validation_loss,
            "system/learning_rate": learning_rate
        }
        mlflow.log_metrics(metrics, step=epoch)

    def log_training_epoch_data(self, epoch, bidx, batch_count, avg_training_loss):
        live_message = f"[Training] Epoch: {epoch} | Batch: {bidx}/{batch_count} | Loss: {avg_training_loss:.4f}"
        print(f"\r{live_message}", end="")

        mlflow.set_tag("State", "Training")
        mlflow.set_tag("Epoch", epoch)
        mlflow.set_tag("Batch", f"{bidx}/{int(batch_count)}")
        mlflow.set_tag("Avg Epoch Training Loss", round(avg_training_loss, 5))

    def log_validation_epoch_data(self, epoch, bidx, batch_count, avg_validation_loss):
        live_message = f"[Validating] Epoch: {epoch} | Batch: {bidx}/{batch_count} | Loss: {avg_validation_loss:.4f}"
        print(f"\r{live_message}", end="")

        mlflow.set_tag("State", "Validating")
        mlflow.set_tag("Epoch", epoch)
        mlflow.set_tag("Batch", f"{bidx}/{int(batch_count)}")
        mlflow.set_tag("Avg Epoch Validation Loss", round(avg_validation_loss, 5))

    def log_test_text(self, prompts: List[Tuple[str, str]], epoch):
        artifact_path = f"text_logs/epoch_{epoch}"
        # mlflow.log_text(message, f"{artifact_path}/{title}.txt")
        full_note = f"# Epoch {epoch}\n\n"
        for title, message in prompts:
            full_note += f"### {title}\n{message}\n\n"
        mlflow.set_tag("mlflow.note.content", full_note)
