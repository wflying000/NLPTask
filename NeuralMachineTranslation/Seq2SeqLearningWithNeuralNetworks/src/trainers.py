import os
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        eval_dataloader,
        compute_loss,
        compute_metrics,
        outputs_dir,
        training_args,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_loss = compute_loss
        self.compute_metrics = compute_metrics
        self.training_args = training_args
        self.outputs_dir = outputs_dir
        self.writer = SummaryWriter(outputs_dir)
    
    def train(self):
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        training_args = self.training_args
        tokenizer = training_args.tokenizer
        tgt_new2old = training_args.tgt_new2old
        best_loss = 1e9

        device = None
        for _, p in model.named_parameters():
            device = p.device
            break
        
        total_loss = 0
        for epoch in range(training_args.num_epochs):
            model.train()
            epoch_loss = 0

            predictions = []
            references = []
            for batch_idx, batch in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
                inputs = {k: v.to(device) for k, v in batch.items() if k in training_args.cuda_item}

                outputs = model(inputs)

                logits = outputs["logits"]
                labels = inputs["labels"]

                loss = self.compute_loss(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                
                total_loss += loss.item()
                epoch_loss += loss.item()

                # cur_preds = model.decode(logits.detach().cpu(), tokenizer, tgt_new2old)
                # cur_refs = [[x] for x in batch["tgt_text"]]
                # predictions += cur_preds
                # references += cur_refs

                global_step = len(self.train_dataloader) * epoch + batch_idx
                if (global_step + 1) % training_args.write_step == 0:
                    avg_loss = total_loss / (global_step + 1)
                    self.writer.add_scalar("Train-Step-Loss", avg_loss, global_step=global_step)
            
            epoch_loss /= len(self.train_dataloader)
            # train_metrics = self.compute_loss(predictions=predictions, references=references)
            # train_sacrebleu_score = train_metrics["score"]
            self.writer.add_scalar("Train-Epoch-Loss", epoch_loss, global_step=epoch)
            # self.writer.add_scalar("Train-Epoch-Score", train_sacrebleu_score, global_step=epoch)


            eval_results = self.evaluate()
            eval_loss = eval_results["loss"]
            eval_metrics = eval_results["metrics"]
            eval_sacrebleu_score = eval_metrics["score"]
            self.writer.add_scalar("Eval-Epoch-Loss", eval_loss, global_step=epoch)
            self.writer.add_scalar("Eval-Epoch-Score", eval_sacrebleu_score, global_step=epoch)

            if eval_loss < best_loss:
                save_path = os.path.join(self.outputs_dir, "model.pth")
                torch.save(model.state_dict(), save_path)
                best_loss = eval_loss

    def evaluate(self):
        training_args = self.training_args
        tokenizer = training_args.tokenizer
        tgt_new2old = training_args.tgt_new2old
        model = self.model
        model.eval()
        
        device = None
        for _, p in model.named_parameters():
            device = p.device
            break
        
        eval_loss = 0
        predictions = []
        references = []

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, total=len(self.eval_dataloader)):
                inputs = {k: v.to(device) for k, v in batch.items() if k in training_args.cuda_item}

                outputs = model(inputs)

                logits = outputs["logits"]
                labels = inputs["labels"]
                loss = self.compute_loss(logits, labels)
                eval_loss += loss.item()

                cur_preds = model.decode(logits.detach().cpu(), tokenizer, tgt_new2old)
                cur_refs = [[x] for x in batch["tgt_text"]]
                
                predictions += cur_preds
                references += cur_refs
        
        eval_loss /= len(self.eval_dataloader)
        metrics = self.compute_metrics(predictions=predictions, references=references)

        eval_results = {
            "loss": eval_loss,
            "metrics": metrics,
            "predictions": predictions,
            "references": references,
        }
        
        return eval_results


                    

