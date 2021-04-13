"""
Entry point for training and testing IMU transformers
Code Forked from https://github.com/yolish/transposenet and modified for this purpose
"""
import argparse
import torch
import numpy as np
import json
import logging
from util import utils
from os.path import join
from models.IMUTransformerDenoiser import IMUTransformerDenoiser
from models.IMUCNNDenoiser import IMUCNNDenoiser

from util.IMUDataset import IMUDataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("mode", help="train or test")
    arg_parser.add_argument("imu_dataset_file", help="path to a file mapping imu samples to labels")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained model (should match the model indicated in model_name")
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment/commit used")

    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Start {}ing IMU-transformers".format(args.mode))
    if args.experiment is not None:
        logging.info("Experiment details: {}".format(args.experiment))
    logging.info("Using imu dataset file: {}".format(args.imu_dataset_file))

    # Read configuration
    with open('config.json', "r") as read_file:
        config = json.load(read_file)
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Create the model according to the task
    use_baseline = config.get('use_cnn_baseline')
    if use_baseline:
        model = IMUCNNDenoiser(config).to(device)
    else:
        model = IMUTransformerDenoiser(config).to(device)

    # Load the checkpoint if needed
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    if args.mode == 'train':
        # Set to train mode
        model.train()

        # Set the loss
        loss = torch.nn.MSELoss()

        # Set the optimizer and scheduler
        optim = torch.optim.Adam(model.parameters(),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        # Set the dataset and data loader
        window_size = config.get("window_size")
        input_size = config.get("input_dim")
        dataset = IMUDataset(args.imu_dataset_file, window_size, input_size)
        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")

        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        for epoch in range(n_epochs):

            # Resetting temporal loss used for logging
            running_loss = 0.0
            n_samples = 0

            for batch_idx, minibatch in enumerate(dataloader):
                minibatch["imu"] = minibatch["imu"].to(device).to(dtype=torch.float32)
                label = minibatch.get('label').to(device).to(dtype=torch.float32)
                batch_size = label.shape[0]
                n_samples += batch_size
                n_total_samples += batch_size

                # Zero the gradients
                optim.zero_grad()

                # Forward pass
                res = model(minibatch)

                # Compute loss
                criterion = loss(res, label)

                # Collect for recoding and plotting
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss on train set
                if batch_idx % n_freq_print == 0:
                    logging.info("[Batch-{}/Epoch-{}] running loss: {:.3f}".format(
                                                                        batch_idx+1, epoch+1,
                                                                        (running_loss/n_samples)))
            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        torch.save(model.state_dict(), checkpoint_prefix + '_final.pth'.format(epoch))

        # Plot the loss function
        loss_fig_path = checkpoint_prefix + "_loss_fig.png"
        utils.plot_loss_func(sample_count, loss_vals, loss_fig_path)

    else: # Test
        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        window_size = config.get("window_size")
        input_size = config.get("input_dim")
        dataset = IMUDataset(args.imu_dataset_file, window_size, input_size)
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        metric = []
        metric_orig = []
        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):

                minibatch["imu"] = minibatch["imu"].to(device).to(dtype=torch.float32)
                label = minibatch.get('label').to(device).to(dtype=torch.float32)
                orig = minibatch.get('imu').to(device).to(dtype=torch.float32)

                # Forward pass to predict the pose
                res = model(minibatch)
                plt.plot(res.cpu().numpy()[0,:,0],'-b')
                plt.plot(res.cpu().numpy()[0, :, 1],'-b')
                plt.plot(res.cpu().numpy()[0, :, 2],'-b')
                # plt.show()
                plt.plot(label.cpu().numpy()[0,:,0],'-g')
                plt.plot(label.cpu().numpy()[0, :, 1],'-g')
                plt.plot(label.cpu().numpy()[0, :, 2],'-g')
                # plt.show()
                # Evaluate and append
                curr_metric = torch.norm(res-label)
                orig_metric = torch.norm(orig-label)

                metric.append(curr_metric.item())
                metric_orig.append(orig_metric.item())

        # Record overall statistics
        stats_msg = "Performance of {} on {}".format(args.checkpoint_path, args.imu_dataset_file)

        stats_msg = stats_msg + "\n\tMean Diff Prediction - GT L2: {}".format(np.mean(metric)) # TO DO change as required
        stats_msg = stats_msg + "\n\tMean Diff Undenoised Signal - GT L2: {}".format(np.mean(metric_orig)) # TO DO change as required


        logging.info(stats_msg)

