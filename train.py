import argparse
import datetime
import os

import torch
import torch.optim as optim
from tqdm import tqdm

import dataset.data_loader as data_loader
import model.net as net
from loss.loss import compute_loss, compute_metrics
from common import utils
from common.manager import Manager

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="experiments/base_model", help="Directory containing params.json")
parser.add_argument("--restore_file",
                    default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")
parser.add_argument("-ow", "--only_weights", action="store_true", help="Only use weights to load or load all train status.")


def train(model, manager: Manager):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # loss status initial
    manager.reset_loss_status()

    # set model to training mode
    torch.cuda.empty_cache()
    model.train()

    # Use tqdm for progress bar
    with tqdm(total=len(manager.dataloaders["train"])) as t:
        for batch_idx, data_batch in enumerate(manager.dataloaders["train"]):
            # move to GPU if available
            data_batch = utils.tensor_gpu(data_batch)

            # compute model output and loss
            output_batch = model(data_batch)
            losses = compute_loss(output_batch, manager.params)

            # real batch size
            batch_size = data_batch["points_src"].size()[0]

            # update loss status and print current loss and average loss
            manager.update_loss_status(loss=losses, batch_size=batch_size)

            # clear previous gradients, compute gradients of all variables wrt loss
            manager.optimizer.zero_grad()
            losses["total"].backward()
            # performs updates using calculated gradients
            manager.optimizer.step()

            manager.write_loss_to_tb(split="train")

            # update step: step += 1
            manager.update_step()

            # info print
            print_str = manager.print_train_info()

            t.set_description(desc=print_str)
            t.update()

    manager.scheduler.step()
    # update epoch: epoch += 1
    manager.update_epoch()


def evaluate(model, manager: Manager):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        # compute metrics over the dataset
        if manager.dataloaders["val"] is not None:
            # loss status and val status initial
            manager.reset_loss_status()
            manager.reset_metric_status("val")
            for batch_idx, data_batch in enumerate(manager.dataloaders["val"]):
                # move to GPU if available
                data_batch = utils.tensor_gpu(data_batch)
                # compute model output
                output_batch = model(data_batch)
                # real batch size
                batch_size = data_batch["points_src"].size()[0]
                # compute all loss on this batch
                loss = compute_loss(output_batch, manager.params)
                manager.update_loss_status(loss, batch_size)
                # compute all metrics on this batch
                metrics = compute_metrics(output_batch, manager.params)
                manager.update_metric_status(metrics, "val", batch_size)

            # compute RMSE metrics
            manager.summarize_metric_status(metrics, "val")
            # update data to tensorboard
            manager.write_metric_to_tb(split="val")
            # For each epoch, update and print the metric
            manager.print_metrics("val", title="Val", color="green", only_best=True)

        if manager.dataloaders["test"] is not None:
            # loss status and test status initial
            manager.reset_loss_status()
            manager.reset_metric_status("test")
            for batch_idx, data_batch in enumerate(manager.dataloaders["test"]):
                # move to GPU if available
                data_batch = utils.tensor_gpu(data_batch)
                # compute model output
                output_batch = model(data_batch)
                # real batch size
                batch_size = data_batch["points_src"].size()[0]
                # compute all loss on this batch
                loss = compute_loss(output_batch, manager.params)
                manager.update_loss_status(loss, batch_size)
                # compute all metrics on this batch
                metrics = compute_metrics(output_batch, manager.params)
                manager.update_metric_status(metrics, "test", batch_size)

            # compute RMSE metrics
            manager.summarize_metric_status(metrics, "test")
            # update data to tensorboard
            manager.write_metric_to_tb(split="test")
            # For each epoch, update and print the metric
            manager.print_metrics("test", title="Test", color="red", only_best=True)


def train_and_evaluate(model, manager: Manager):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
    """

    # reload weights from restore_file if specified
    if args.restore_file is not None:
        manager.load_checkpoints()

    for epoch in range(manager.epoch, manager.params.num_epochs):
        # compute number of batches in one epoch (one full pass over the training set)
        train(model, manager)

        # Evaluate for one epoch on validation set
        evaluate(model, manager)

        # Check if current is best, save checkpoints if best, meanwhile, save latest checkpoints
        manager.check_best_save_last_checkpoints(save_latest_freq=100, save_best_after=1000)


if __name__ == "__main__":
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Update args into params
    params.update(vars(args))

    # Set the logger
    logger = utils.set_logger(os.path.join(params.model_dir, "train.log"))

    # Set the tensorboard writer
    log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    if params.cuda:
        num_gpu = torch.cuda.device_count()
        if num_gpu > 0:
            torch.cuda.set_device(0)
        gpu_ids = ", ".join(str(i) for i in [j for j in range(num_gpu)])
        logger.info("Using GPU ids: [{}]".format(gpu_ids))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

    # Define the model and optimizer
    if params.cuda:
        model = net.fetch_net(params).cuda()
        model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    else:
        model = net.fetch_net(params)

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.gamma)

    # initial status for checkpoint manager
    manager = Manager(model=model, optimizer=optimizer, scheduler=scheduler, params=params, dataloaders=dataloaders, logger=logger)

    # Train the model
    logger.info("Starting training for {} epoch(s)".format(params.num_epochs))

    train_and_evaluate(model, manager)
