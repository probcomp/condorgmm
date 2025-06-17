import glob
import condorgmm
import fire
import matplotlib.pyplot as plt
import pickle

results_directory = (
    "/home/nishadgothoskar/condorgmm/results/gsplat_multires_visuals_2__2025-03-09-06-16-24"
)


def generate_gsplat_multires_visuals(results_directory):
    results_files = glob.glob(results_directory + "/*.pkl")
    data = pickle.load(open(results_files[0], "rb"))
    num_gaussians_list = list(data.keys())
    num_gaussians_list.sort()

    # Create a figure with subplots based on number of images
    n = len(num_gaussians_list)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    # If only one image, axes will not be array - convert to array for consistent indexing
    if n == 1:
        axes = [axes]

    # Turn off axes for cleaner visualization
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot each image in its corresponding subplot
    for idx, num_gaussians in enumerate(num_gaussians_list):
        image = data[num_gaussians]["image"] / 255.0
        axes[idx].imshow(image)
        axes[idx].set_title(f"N={num_gaussians}", fontsize=18)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    tex_dir = condorgmm.get_root_path() / "condorgmm_tex"
    filepath = tex_dir / "gsplat_multires_visuals.png"

    # Save the figure
    plt.savefig(filepath, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    fire.Fire(generate_gsplat_multires_visuals)
