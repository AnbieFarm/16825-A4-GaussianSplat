import argparse
import os
import os.path as osp
import random
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F

from nerf.config_parser import add_config_arguments
from nerf.network_grid import NeRFNetwork
from nerf.provider import NeRFDataset
from optimizer import Adan
from PIL import Image
from SDS import SDS
import torchvision.transforms 
from utils import prepare_embeddings, seed_everything
import matplotlib.pyplot as plt

# AH: add loss plots
def save_training_loss_plot(losses, cur_step, save_path):
    avg_interval = 10
    # Compute averages every `avg_interval` steps
    avg_losses = [np.mean(np.array(losses[i:min(i + avg_interval, len(losses) - 1)])) for i in range(0, len(losses), avg_interval)]
    avg_steps = np.array([i for i in range(len(avg_losses))]) * avg_interval
    #print("losses: ", losses)
    #print("avg losses: ", avg_losses)
    #print("avg steps: ", avg_steps)

    plt.figure(figsize=(8, 5))
    plt.plot(avg_steps, avg_losses, label="Training Loss", color="blue")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Averaged over {avg_interval} steps until Step {cur_step}")
    plt.grid(True)

    # Save the plot
    #filename = f"loss_vs_steps_{cur_step}.png"
    #plt.savefig(f"{save_path}/{filename}")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved plot {save_path}")

def optimize_nerf(
    sds,
    prompt,
    neg_prompt="",
    device="cpu",
    log_interval=4,
    args=None,
):
    """
    Optimize the view for a NeRF model to match the prompt.
    """

    # Step 1. Create text embeddings from prompt
    embeddings = prepare_embeddings(sds, prompt, neg_prompt, view_dependent=False)

    # Step 2. Set up NeRF model
    model = NeRFNetwork(args).to(device)

    # Step 3. Create optimizer and training parameters
    lr = 1e-3
    optimizer = Adan(
        model.parameters(),
        lr=5 * lr,
        eps=1e-8,
        weight_decay=2e-5,
        max_grad_norm=5.0,
        foreach=False,
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1)
    if args.loss_scaling:
        scaler = torch.cuda.amp.GradScaler()

    # Step 4. Load the dataset
    train_loader = NeRFDataset(
        args,
        device=device,
        type="train",
        H=args.h,
        W=args.w,
        size=args.dataset_size_train * args.batch_size,
    ).dataloader()
    test_loader = NeRFDataset(
        args,
        device=device,
        type="test",
        H=args.h,
        W=args.w,
        size=args.dataset_size_test,
    ).dataloader(batch_size=1)

    # Step 5. Training loop
    epoch_loss_dict = {} # record 1 loss per epoch
    step_loss_dict = {} # record the loss for each step
    global_step = 0
    # create logging and saving directories
    os.makedirs(f"{sds.output_dir}/images", exist_ok=True)
    os.makedirs(f"{sds.output_dir}/videos", exist_ok=True)
    os.makedirs(f"{sds.output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{sds.output_dir}/loss_graphs", exist_ok=True)
    checkpoint_path = osp.join(sds.output_dir, f"checkpoints/nerf_checkpoint")

    max_epoch = np.ceil(args.iters / len(train_loader)).astype(np.int32)

    for epoch in range(max_epoch):
        model.train()
        for data in train_loader:
            global_step += 1

            # Initialize optimizer
            optimizer.zero_grad()
            # experiment iterations ratio
            # i.e. what proportion of this experiment have we completed (in terms of iterations) so far?
            exp_iter_ratio = (global_step - args.exp_start_iter) / (
                args.exp_end_iter - args.exp_start_iter
            )

            # Load the data
            rays_o = data["rays_o"]  # [B, N, 3]
            rays_d = data["rays_d"]  # [B, N, 3]
            mvp = data["mvp"]  # [B, 4, 4]

            B, N = rays_o.shape[:2]
            H, W = data["H"], data["W"]
            assert B == 1, "Batch size should be 1"

            # When ref_data has B images > args.batch_size
            if B > args.batch_size:
                # choose batch_size images out of those B images
                choice = torch.randperm(B)[: args.batch_size]
                B = args.batch_size
                rays_o = rays_o[choice]
                rays_d = rays_d[choice]
                mvp = mvp[choice]

            # Set the shading and background color for rendering
            if exp_iter_ratio <= args.latent_iter_ratio:
                ambient_ratio = 1.0
                shading = "normal"
                bg_color = None

            else:
                # random shading
                ambient_ratio = (
                    args.min_ambient_ratio
                    + (1.0 - args.min_ambient_ratio) * random.random()
                )
                rand = random.random()
                if rand >= (1.0 - args.textureless_ratio):
                    shading = "textureless"
                else:
                    shading = "lambertian"

                # random background
                rand = random.random()
                if args.bg_radius > 0 and rand > 0.5:
                    bg_color = None  # use bg_net
                else:
                    bg_color = torch.rand(3).to(device)  # single color random bg

            # Forward pass to render NeRF model
            outputs = model.render(
                rays_o,
                rays_d,
                mvp,
                H,
                W,
                staged=False,
                perturb=True,
                bg_color=bg_color,
                ambient_ratio=ambient_ratio,
                shading=shading,
                binarize=False,
                max_ray_batch=args.max_ray_batch,
            )

            # TODO AH: Extra debug stuff!!!!!
            if global_step % 100 == 0:
                print(f"Mean alpha (opacity): {outputs['weights'].mean().item():.6f}")

            pred_rgb = (
                outputs["image"].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
            )  # [B, 3, H, W]

            # Compuate the loss
            # interpolate text_z
            azimuth = data["azimuth"]  # [-180, 180]
            assert azimuth.shape[0] == 1, "Batch size should be 1"
            text_uncond = embeddings["uncond"]

            if not args.view_dep_text:
                text_cond = embeddings["default"]
            else:
                #print("Encoding text with view dependency!!")
                # Create text embedding with view dependency.
                embeddings_w_view = prepare_embeddings(sds, prompt, neg_prompt, view_dependent=True)

                if -45 <= azimuth <= 45:
                    text_cond = embeddings_w_view['front']
                elif azimuth <=-135 or azimuth >= 135:
                    text_cond = embeddings_w_view['back']
                else:
                    text_cond = embeddings_w_view['side']
  
            ### YOUR CODE HERE ###
            pred_rgb = F.interpolate(pred_rgb, size=(512, 512), mode='bilinear', align_corners=False)

            # Compute SDS loss either in pixel space or latent space
            if args.sds_pixel_space:
                loss = sds.sds_loss(pred_rgb, text_cond, text_embeddings_uncond=text_uncond, in_pixel_space=True)
            else:
                latents = sds.encode_imgs(pred_rgb)
                loss = sds.sds_loss(latents, text_cond, text_embeddings_uncond=text_uncond, in_pixel_space=False)

            # regularizations
            if args.lambda_entropy > 0:
                alphas = outputs["weights"].clamp(1e-5, 1 - 1e-5)
                # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
                loss_entropy = (
                    -alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)
                ).mean()
                lambda_entropy = args.lambda_entropy * min(
                    1, 2 * global_step / args.iters
                )
                loss = loss + lambda_entropy * loss_entropy

            if args.lambda_orient > 0 and "loss_orient" in outputs:
                loss_orient = outputs["loss_orient"]
                loss = loss + args.lambda_orient * loss_orient

            # Backward pass
            if args.loss_scaling:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            lr_scheduler.step()

            # Log
            step_loss_dict[global_step] = loss.item()
            print(f"Epoch {epoch}, global_step {global_step}, loss {loss.item()}")
            if global_step % 100 == 0:
                epoch_loss_dict[global_step] = loss.item()
                # save the nerf rendering as the logging output, instead of the decoded latent
                imgs = (
                    pred_rgb.detach().cpu().permute(0, 2, 3, 1).numpy()
                )  # torch to numpy, shape [1, 512, 512, 3]
                imgs = (imgs * 255).round()  # [0, 1] => [0, 255]
                rgb = Image.fromarray(imgs[0].astype("uint8"))
                output_path = (
                    f"{sds.output_dir}/images/rgb_epoch_{epoch}_iter_{global_step}.png"
                )
                rgb.save(output_path)

        # Save checkpoint
        if epoch % log_interval == 0 and global_step > 0:
            cur_checkpoint_path = checkpoint_path + f"_{epoch}.pth"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                cur_checkpoint_path,
            )
            print(f"-- Saved checkpiont to {cur_checkpoint_path}!")

        # Save loss plot.
        if epoch % log_interval == 0 and global_step > 10:
             save_path = osp.join(sds.output_dir, f"loss_graphs/nerf_training_loss_epoch{epoch}_step{global_step}.png")
             save_training_loss_plot(list(step_loss_dict.values()), global_step, save_path)

        # Save test prediction for this epoch
        if epoch % log_interval == 0 or epoch == max_epoch - 1:
            model.eval()
            all_preds = []
            all_preds_depth = []

            print(f"Epoch {epoch}, testing and save rgb and depth to video...")

            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    rays_o = data["rays_o"]  # [B, N, 3]
                    rays_d = data["rays_d"]  # [B, N, 3]
                    mvp = data["mvp"]

                    B, N = rays_o.shape[:2]
                    H, W = data["H"], data["W"]

                    if bg_color is not None:
                        bg_color = bg_color.to(rays_o.device)

                    shading = data["shading"] if "shading" in data else "albedo"
                    ambient_ratio = (
                        data["ambient_ratio"] if "ambient_ratio" in data else 1.0
                    )
                    light_d = data["light_d"] if "light_d" in data else None

                    outputs = model.render(
                        rays_o,
                        rays_d,
                        mvp,
                        H,
                        W,
                        staged=True,
                        perturb=False,
                        light_d=light_d,
                        ambient_ratio=ambient_ratio,
                        shading=shading,
                        bg_color=bg_color,
                    )

                    preds = outputs["image"].reshape(B, H, W, 3)
                    preds_depth = outputs["depth"].reshape(B, H, W)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth - pred_depth.min()) / (
                        pred_depth.max() - pred_depth.min() + 1e-6
                    )
                    pred_depth = (pred_depth * 255).astype(np.uint8)

                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            # save the video
            '''imageio.mimwrite(
                os.path.join(sds.output_dir, "videos", f"rgb_ep_{epoch}.mp4"),
                all_preds,
                fps=25,
                quality=8,
                macro_block_size=1,
            )
            imageio.mimwrite(
                os.path.join(sds.output_dir, "videos", f"depth_ep_{epoch}.mp4"),
                all_preds_depth,
                fps=25,
                quality=8,
                macro_block_size=1,
            )'''
            imageio.mimwrite(
                os.path.join(sds.output_dir, "videos", f"rgb_ep_{epoch}.gif"),
                all_preds,
                fps=25,
                quality=8,
                macro_block_size=1,
            )
            imageio.mimwrite(
                os.path.join(sds.output_dir, "videos", f"depth_ep_{epoch}.gif"),
                all_preds_depth,
                fps=25,
                quality=8,
                macro_block_size=1,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a vase with pink flowers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--loss_scaling", type=int, default=1)

    ### YOUR CODE HERE ###
    # You wil need to tune the following parameters to obtain good NeRF results
    ### regularizations
    parser.add_argument('--lambda_entropy', type=float, default=1e-3, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_orient', type=float, default=1e-2, help="loss scale for orientation")

    ### shading options
    parser.add_argument('--latent_iter_ratio', type=float, default=0.2, help="training iters that only use albedo shading")

    ### AH: add option to run SDS loss in pixel space!
    parser.add_argument('--pixel_space_loss', type=float, int=0, help="if True then SDS loss is computed in pixel space not latent space")

    parser.add_argument(
        "--postfix", type=str, default="_[with_view]", help="Postfix for the output directory"
    )
    parser.add_argument(
        "--view_dep_text",
        type=int,
        default=0,
        help="option to use view dependent text embeddings for nerf optimization",
    )
    parser = add_config_arguments(
        parser
    )  # add additional arguments for nerf optimization, You don't need to change the setting here by default

    args = parser.parse_args()

    seed_everything(args.seed)

    # create output directory
    args.output_dir = osp.join(args.output_dir, "nerf")
    output_dir = os.path.join(
        args.output_dir, args.prompt.replace(" ", "_") + args.postfix
    )
    os.makedirs(output_dir, exist_ok=True)

    # initialize SDS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sds = SDS(sd_version="2.1", device=device, output_dir=output_dir)

    # optimize a NeRF model
    start_time = time.time()
    optimize_nerf(sds, prompt=args.prompt, device=device, args=args)
    print(f"Optimization took {time.time() - start_time:.2f} seconds")
