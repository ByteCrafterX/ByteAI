# train_lora.py
# Exemplo de treinamento DreamBooth + LoRA com Diffusers (estilo oficial)
# =====================================================
import argparse
import logging
import math
import os

import torch
import torch.utils.checkpoint
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_scheduler
# (Removemos AttnProcsLayers, pois agora injetamos LoRA manualmente)
# from diffusers.loaders import AttnProcsLayers

import itertools
import random

# ========== ADAPTADO: CONFIGURA LOG PARA ARQUIVO EXTERNO ==========
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

log_file = "train_lora.log"
file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info("===== Iniciando script train_lora.py (estilo DreamBooth + LoRA) =====")


# -------------------------------------------------
# 1) Dataset
# -------------------------------------------------
class ImageDataset(Dataset):
    """
    Dataset simples: lê todas as imagens de train_data_dir
    e usa a prompt "instance_prompt" para cada uma (estilo DreamBooth).
    """
    def __init__(self, data_root, instance_prompt, size=512):
        self.data_root = data_root
        self.instance_prompt = instance_prompt
        self.size = size
        self.images_paths = []

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for root, dirs, files in os.walk(data_root):
            for f in files:
                if f.lower().endswith(exts):
                    self.images_paths.append(os.path.join(root, f))

        self.num_images = len(self.images_paths)
        if self.num_images == 0:
            raise ValueError(f"Nenhuma imagem encontrada em {data_root}.")

        self.transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        image_path = self.images_paths[index % self.num_images]
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)

        example = {
            "instance_images": image,
            "instance_prompt": self.instance_prompt
        }
        return example


# -------------------------------------------------
# 2) Definição de Argumentos
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Modelo base (Stable Diffusion) do Hugging Face.")
    parser.add_argument("--revision", type=str, default=None,
                        help="Se o modelo precisar de revision (ex: branch ou tag).")
    parser.add_argument("--train_data_dir", type=str, default=None,
                        help="Diretório com as imagens de treino.")
    parser.add_argument("--instance_prompt", type=str, default="",
                        help="Prompt de instância (DreamBooth).")
    parser.add_argument("--output_dir", type=str, default="./lora_output",
                        help="Onde salvar o LoRA final.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed (opcional) para reprodutibilidade.")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Tamanho das imagens (512 ou 768).")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help="constant ou constant_with_warmup ou cosine, etc.")
    parser.add_argument("--lr_warmup_steps", type=int, default=0,
                        help="Steps de aquecimento no scheduler.")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    args = parser.parse_args()
    return args


# -------------------------------------------------
# 3) Função para Injetar LoRA nas camadas to_q, to_k, to_v, to_out.0
# -------------------------------------------------
def patch_lora_layers(unet, rank=4, alpha=1.0):
    """
    Modo oficial DreamBooth + LoRA:
    Insere LoRA manualmente nas camadas de atenção do UNet (to_q, to_k, to_v, to_out.0).
    """
    lora_parameters = []

    # Subclasse simples de nn.Module que implementa a projeção LoRA
    class LoRALinear(nn.Module):
        def __init__(self, orig_linear, rank=4, alpha=1.0):
            super().__init__()
            self.orig = orig_linear
            self.rank = rank
            self.scaling = alpha / rank

            # Nova projeção LoRA
            in_dim = orig_linear.in_features
            out_dim = orig_linear.out_features
            self.lora_down = nn.Linear(in_dim, rank, bias=False)
            self.lora_up = nn.Linear(rank, out_dim, bias=False)

            # Inicializa
            nn.init.zeros_(self.lora_down.weight)
            nn.init.zeros_(self.lora_up.weight)

        def forward(self, x):
            # Saída normal
            result = self.orig(x)
            # Acrescenta LoRA
            lora = self.lora_up(self.lora_down(x)) * self.scaling
            return result + lora

    # Itera sobre todos submódulos do unet
    for name, module in unet.named_modules():
        if any(x in name for x in [".to_q", ".to_k", ".to_v", ".to_out.0"]):
            if isinstance(module, nn.Linear):
                # Substitui o module por LoRALinear
                wrapped = LoRALinear(module, rank=rank, alpha=alpha)
                # injetamos:
                parent = unet
                # Precisamos achar o "parent" e a "atribuição" exata
                *parent_path, child_name = name.split(".")
                for p_name in parent_path:
                    parent = getattr(parent, p_name)
                setattr(parent, child_name, wrapped)
                # Guardamos referência aos pesos
                lora_parameters.append(wrapped.lora_down.weight)
                lora_parameters.append(wrapped.lora_up.weight)

    return lora_parameters


# -------------------------------------------------
# 4) collate_fn
# -------------------------------------------------
def collate_fn(examples, tokenizer):
    pixel_values = [ex["instance_images"] for ex in examples]
    prompts = [ex["instance_prompt"] for ex in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    return {
        "pixel_values": pixel_values,
        "input_ids": text_inputs.input_ids,
        "attention_mask": text_inputs.attention_mask,
    }


# -------------------------------------------------
# 5) Criação do pipeline e do loop de treinamento
# -------------------------------------------------
def train_loop(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16" if torch.cuda.is_available() else "no"
    )
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
    logger.info(accelerator.state)
    set_seed(args.seed)

    logger.info("Criando dataset...")
    train_dataset = ImageDataset(
        data_root=args.train_data_dir,
        instance_prompt=args.instance_prompt,
        size=args.resolution
    )

    # Loader inicial (só p/ ver dataset)
    initial_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    logger.info(f"Total de imagens: {len(train_dataset)}")

    # Carrega pipeline base
    logger.info(f"Carregando pipeline base: {args.pretrained_model_name}")
    pipe = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name,
        revision=args.revision,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    unet = pipe.unet
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # Congele tudo
    for p in unet.parameters():
        p.requires_grad = False
    for p in vae.parameters():
        p.requires_grad = False
    for p in text_encoder.parameters():
        p.requires_grad = False

    # Injetar LoRA
    logger.info("Injetando LoRA nas camadas to_q, to_k, to_v, to_out.0...")
    lora_params = patch_lora_layers(unet, rank=4, alpha=1.0)
    logger.info(f"Total de parâmetros LoRA (treináveis): {sum(p.numel() for p in lora_params)}")

    # Define DataLoader real (com tokenizer) para o loop
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda ex: collate_fn(ex, tokenizer),
    )

    # Otimizador
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps
    )

    # Acelera
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    vae.to(accelerator.device, dtype=unet.dtype)
    text_encoder.to(accelerator.device, dtype=unet.dtype)

    global_step = 0
    for epoch in range(9999):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 1) Texto
                input_ids = batch["input_ids"].to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids)[0]

                # 2) Imagens -> latents
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=unet.dtype)
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

                # 3) Adiciona ruído e timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device).long()
                noisy_latents = latents + noise  # simplificado

                # 4) UNet prediz o ruído
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # 5) Loss = MSE do ruído
                loss = torch.nn.functional.mse_loss(model_pred, noise, reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            if global_step % 50 == 0:
                logger.info(f"Step {global_step}/{args.max_train_steps}, loss={loss.item():.4f}")

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.print("Treino finalizado. Salvando LoRA...")

    accelerator.wait_for_everyone()
    unet = accelerator.unwrap_model(unet)

    # Agora, salvamos os pesos do UNet contendo as projeções LoRA injetadas
    os.makedirs(args.output_dir, exist_ok=True)
    # Tradicionalmente, salva o UNet inteiro modificado + text_encoder se quiser
    # Mas, no DreamBooth LoRA oficial, costuma-se extrair só os deltas. Aqui salvaremos o unet completo
    unet.save_pretrained(args.output_dir)

    accelerator.print(f"LoRA (UNet modificado) salvo em: {args.output_dir}")

    # Opcional: um README
    with open(os.path.join(args.output_dir, "lora_README.txt"), "w") as f:
        f.write(
            f"Treinado com LoRA (rank=4) injetado manualmente.\n"
            f"Steps={args.max_train_steps}, LR={args.learning_rate}\n"
        )


def main():
    args = parse_args()
    if args.train_data_dir is None or not os.path.exists(args.train_data_dir):
        raise ValueError(f"train_data_dir inválido: {args.train_data_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    train_loop(args)


if __name__ == "__main__":
    main()
