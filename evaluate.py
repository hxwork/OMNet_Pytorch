import argparse
import os
from collections import defaultdict

import time
import torch

import dataset.data_loader as data_loader
import model.net as net
from common import utils
from common.manager import Manager
from loss.loss import compute_loss, compute_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="./experiments/experiment_omnet", help="Directory containing params.json")
parser.add_argument("--restore_file", type=str, help="name of the file in --model_dir containing weights to load")


def test(model, manager):
    # set model to evaluation mode
    torch.cuda.empty_cache()
    model.eval()

    with torch.no_grad():
        # compute metrics over the dataset
        if manager.dataloaders["val"] is not None:
            # inference time
            total_time = 0.
            all_endpoints = defaultdict(list)
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
            # For each epoch, update and print the metric
            manager.print_metrics("val", title="Val", color="green")

        if manager.dataloaders["test"] is not None:
            # inference time
            total_time = {"total": 0.}
            total_time_outside = 0.
            all_endpoints = defaultdict(list)
            # loss status and test status initial
            manager.reset_loss_status()
            manager.reset_metric_status("test")
            for batch_idx, data_batch in enumerate(manager.dataloaders["test"]):
                # move to GPU if available
                data_batch = utils.tensor_gpu(data_batch)
                # compute model output
                start_time = time.time()
                output_batch = model(data_batch)
                total_time_outside += time.time() - start_time

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
            # For each epoch, print the metric
            manager.print_metrics("test", title="Test", color="red")


if __name__ == '__main__':
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Only load model weights
    params.only_weights = True

    # Update args into params
    params.update(vars(args))

    # Get the logger
    logger = utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Use GPU if available
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

    # Fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

    # Define the model and optimizer
    if params.cuda:
        model = net.fetch_net(params).cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    else:
        model = net.fetch_net(params)

    # Initial status for checkpoint manager
    manager = Manager(model=model, optimizer=None, scheduler=None, params=params, dataloaders=dataloaders, logger=logger)

    # Reload weights from the saved file
    manager.load_checkpoints()

    # Test the model
    logger.info("Starting test")

    # Evaluate
    test(model, manager)
