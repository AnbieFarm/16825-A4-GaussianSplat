import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from diffusers import DDIMScheduler, StableDiffusionPipeline
#import lpips


class SDS:
    """
    Class to implement the SDS loss function.
    """

    def __init__(
        self,
        sd_version="2.1",
        device="cpu",
        t_range=[0.02, 0.98],
        output_dir="output",
    ):
        """
        Load the Stable Diffusion model and set the parameters.

        Args:
            sd_version (str): version for stable diffusion model
            device (_type_): _description_
        """

        # Set the stable diffusion model key based on the version
        if sd_version == "2.1":
            sd_model_key = "stabilityai/stable-diffusion-2-1-base"
        else:
            raise NotImplementedError(
                f"Stable diffusion version {sd_version} not supported"
            )

        # Set parameters
        self.H = 512  # default height of Stable Diffusion
        self.W = 512  # default width of Stable Diffusion
        self.num_inference_steps = 50
        self.output_dir = output_dir
        self.device = device
        self.precision_t = torch.float32

        # Create model
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            sd_model_key, torch_dtype=self.precision_t
        ).to(device)

        self.vae = sd_pipe.vae
        self.tokenizer = sd_pipe.tokenizer
        self.text_encoder = sd_pipe.text_encoder
        self.unet = sd_pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(
            sd_model_key, subfolder="scheduler", torch_dtype=self.precision_t
        )
        del sd_pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(
            self.device
        )  # for convenient access

        print(f"[INFO] loaded stable diffusion!")

        # Add Perpetual loss fn
        #self.lpips_loss_fn = lpips.LPIPS(net='alex').to('cpu')

    @torch.no_grad()
    def get_text_embeddings(self, prompt):
        """
        Get the text embeddings for the prompt.

        Args:
            prompt (list of string): text prompt to encode.
        """
        # print('here')
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
            #padding=True, # TODO: AH: can we remove this line?
        )
        
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embeddings

    def encode_imgs(self, img):
        """
        Encode the image to latent representation.

        Args:
            img (tensor): image to encode. shape (N, 3, H, W), range [0, 1]

        Returns:
            latents (tensor): latent representation. shape (1, 4, 64, 64)
        """
        # check the shape of the image should be 512x512
        assert img.shape[-2:] == (512, 512), "Image shape should be 512x512"

        img = 2 * img - 1  # [0, 1] => [-1, 1]

        posterior = self.vae.encode(img).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def decode_latents(self, latents):
        """
        Decode the latent representation into RGB image.

        Args:
            latents (tensor): latent representation. shape (1, 4, 64, 64), range [-1, 1]

        Returns:
            imgs[0] (np.array): decoded image. shape (512, 512, 3), range [0, 255]
        """
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents.type(self.precision_t)).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)  # [-1, 1] => [0, 1]
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()  # torch to numpy
        imgs = (imgs * 255).round()  # [0, 1] => [0, 255]
        return imgs[0]


    def sds_loss(
        self,
        latents, # latents 
        text_embeddings,
        text_embeddings_uncond=None,
        guidance_scale=100,
        grad_scale=1,
        pixel_space_loss=0,
        pred_rgb=None,
    ):
        """
        Compute the SDS loss.

        Args:
            latents (tensor): input latents, shape [1, 4, 64, 64]
            text_embeddings (tensor): conditional text embedding (for positive prompt), shape [1, 77, 1024]
            text_embeddings_uncond (tensor, optional): unconditional text embedding (for negative prompt), shape [1, 77, 1024]. Defaults to None.
            guidance_scale (int, optional): weight scaling for guidance. Defaults to 100.
            grad_scale (int, optional): gradient scaling. Defaults to 1.

        Returns:
            loss (tensor): SDS loss
        """

        # sample a timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            (latents.shape[0],),
            dtype=torch.long,
            device=self.device,
        )

        self.scheduler.set_timesteps(self.num_inference_steps)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            ### YOUR CODE HERE ###
            # 1. Add noise to latent space
            std_normal_noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, std_normal_noise, t)

            # 2. Use unet to predict/recover noise added while conditioning on the text prompt 
            pred_noise_cond = self.unet(latents_noisy, t, encoder_hidden_states=text_embeddings).sample
 
            if text_embeddings_uncond is not None and guidance_scale != 1:
                ### YOUR CODE HERE ###
                # 3. (optional:) Bias the predicted noise towards the conditioning.
                pred_noise_uncond = self.unet(latents_noisy, t, encoder_hidden_states=text_embeddings_uncond).sample
                pred_noise = pred_noise_cond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
            else:
                pred_noise = pred_noise_cond
 
        if not pixel_space_loss:
            # 4. Compute noise residual 
            noise_residual = pred_noise - std_normal_noise

            # Compute SDS loss
            # 5. Compute log-likelihood gradient using noise residual
            w = 1 - self.alphas[t]
            log_p_grad = w * noise_residual

            # 6. Create proxy loss that makes the gradient equal to the distillation score
            target_latents = (latents - torch.nan_to_num(grad_scale * log_p_grad)).detach()

            # 7. Use different loss fn depending on whether SDS loss is computed in pixel space or latent space. 
            #loss = MSELoss(reduction='sum')(latents, target_latents) / latents.shape[0]
            loss = F.mse_loss(target_latents, latents)
        else:
            # 4. Denoise latents using the predicted (guided) noise
            denoised_latents = self.scheduler.step(pred_noise, t, latents_noisy).pred_original_sample

            # 5. Decode the denoised latents
            decoded_denoised_rgb = self.decode_latents(denoised_latents.detach())

            # 6. Compute loss between the pred_rgb and the decoded_denoised_rgb
                    # Use AMP for LPIPS to save memory
            #torch.cuda.empty_cache() 
            #with torch.cuda.amp.autocast():
            #    loss = self.lpips_loss_fn(decoded_denoised_rgb, pred_rgb).mean()
            loss = F.mse_loss(decoded_denoised_rgb, pred_rgb)

        return loss
