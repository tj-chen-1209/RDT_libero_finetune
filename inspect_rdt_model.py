"""
Inspect RDT Model Structure - Print all named modules
"""
import torch
import yaml
from models.rdt_runner import RDTRunner


def print_separator(title=""):
    """Print a nice separator"""
    print("\n" + "="*80)
    if title:
        print(f"  {title}")
        print("="*80)


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_structure():
    """Print RDT model structure and module names"""

    # Load config
    print_separator("Loading Configuration")
    with open('configs/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded from configs/base.yaml")
    print(f"  State dim: {config['common']['state_dim']}")
    print(f"  Action chunk size: {config['common']['action_chunk_size']}")

    # Option 1: Load from pretrained (if available)
    print_separator("Loading RDT Model")
    try:
        print("Attempting to load pretrained model...")
        rdt = RDTRunner.from_pretrained(
            "robotics-diffusion-transformer/rdt-1b")
        print("✓ Loaded pretrained model: rdt-1b")
    except Exception as e:
        print(f"Cannot load pretrained model: {e}")
        print("\nCreating model from scratch...")
        rdt = RDTRunner(
            action_dim=config["common"]["state_dim"],
            pred_horizon=config["common"]["action_chunk_size"],
            config=config["model"],
            lang_token_dim=config["model"]["lang_token_dim"],
            img_token_dim=config["model"]["img_token_dim"],
        )
        print("✓ Created model from config")

    # Count parameters
    print_separator("Model Statistics")
    total_params, trainable_params = count_parameters(rdt)
    print(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(
        f"Trainable parameters: {trainable_params:,} ({trainable_params/1e9:.2f}B)")

    # Print all named modules
    print_separator("All Named Modules (Full Hierarchy)")
    print(f"{'Module Name':<80} {'Type':<50}")
    print("-" * 130)

    for name, module in rdt.named_modules():
        module_type = type(module).__name__
        if name:  # Skip root module (empty name)
            print(f"{name:<80} {module_type:<50}")

    # Print top-level modules only
    print_separator("Top-Level Modules")
    print(f"{'Module Name':<40} {'Type':<50} {'Params':<20}")
    print("-" * 110)

    for name, module in rdt.named_children():
        module_type = type(module).__name__
        num_params = sum(p.numel() for p in module.parameters())
        print(f"{name:<40} {module_type:<50} {num_params:>15,} ({num_params/1e6:.1f}M)")

    # Print model architecture summary
    print_separator("Model Architecture Summary")
    print(rdt)

    # Save to file
    print_separator("Saving Results")
    output_file = "rdt_model_structure.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RDT Model Structure - All Named Modules\n")
        f.write("="*80 + "\n\n")

        f.write(
            f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)\n")
        f.write(
            f"Trainable parameters: {trainable_params:,} ({trainable_params/1e9:.2f}B)\n\n")

        f.write("="*80 + "\n")
        f.write("All Named Modules:\n")
        f.write("="*80 + "\n")
        for name, module in rdt.named_modules():
            if name:
                f.write(f"{name} -> {type(module).__name__}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("Model Architecture:\n")
        f.write("="*80 + "\n")
        f.write(str(rdt))

    print(f"✓ Full results saved to: {output_file}")

    print_separator("Complete!")


if __name__ == "__main__":
    try:
        print_model_structure()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
