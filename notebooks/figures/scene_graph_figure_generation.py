import condorgmm
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import networkx as nx
import numpy as np
import matplotlib as mpl

plt.rcParams['font.family'] = 'Times New Roman'


def draw_graph(ax, edge_strength):
    NODE_SIZE = 600  # Reduced node size
    NODE_DIST = 1.0  # Reduced distance between nodes
    pos = {"table": (0, -NODE_DIST), "cup": (0, NODE_DIST)}
    node_colors = {"table": "#f0f0f0", "cup": "#ffcccb"}
    edge_color = mpl.colormaps["Greys"](edge_strength)
    G = nx.DiGraph()
    G.add_node("table")
    G.add_node("cup")
    G.add_edge("table", "cup")

    for node, color in node_colors.items():
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=[node],
            node_shape="o",
            node_size=NODE_SIZE,
            node_color=color,
        )
    if edge_strength > 0.5:
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color=edge_color,
            ax=ax,
            arrowstyle="-|>",
            arrowsize=15,  # Reduced arrow size
            node_size=NODE_SIZE,
            width=1.5,  # Reduced line width
        )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")  # Reduced font size



condorgmm_bucket_dir = condorgmm.get_root_path() / "assets/condorgmm_bucket"
# Save results dictionary to pickle file
results_path = condorgmm_bucket_dir / "scene_graph_results.pkl"
with open(results_path, "rb") as f:
    results = pickle.load(f)
    

key_frames = results["key_frames"]
subsampled_frame_idxs = results["subsampled_frame_idxs"]
N = len(key_frames)


fig = plt.figure(figsize=(15, 8))
gs = GridSpec(4, N, figure=fig, height_ratios=[3, 1.5, 3, 2])  # Increased second row height

fig_axes = [[fig.add_subplot(gs[i, j]) for j in range(N)] for i in range(3)]

plot_ax = fig.add_subplot(gs[3, :])
# plot_ax.vlines(KEY_FRAMES, 0, 1)

floating_probs = results["floating_probs"]
on_table_probs = results["on_table_probs"]
plot_ax.plot(floating_probs, label="Probability of Floating", linewidth=3)
plot_ax.plot(on_table_probs, label="Probability of In Contact", linewidth=3)
plot_ax.set_xlabel("Frame", fontsize=22)
plot_ax.set_ylabel("Probability", fontsize=22)
plot_ax.tick_params(axis='both', which='major', labelsize=18)
plot_ax.set_yticks([0.0, 0.5, 1.0])
plot_ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.9), fontsize=18, ncol=2)
plot_ax.set_ylim(0.0, 1.0)
plot_ax.set_xlim(min(key_frames), max(key_frames))




video_data = results["video_data"]
for i, fidx in enumerate(key_frames):
    fig_axes[0][i].imshow(video_data[i])
    fig_axes[0][i].set_xticks([])
    fig_axes[0][i].set_yticks([])
    fig_axes[0][i].spines['top'].set_visible(False)
    fig_axes[0][i].spines['bottom'].set_visible(False)
    fig_axes[0][i].spines['left'].set_visible(False)
    fig_axes[0][i].spines['right'].set_visible(False)
    
    fig_axes[2][i].set_xticks([])
    fig_axes[2][i].set_yticks([])
    fig_axes[2][i].spines['top'].set_visible(False)
    fig_axes[2][i].spines['bottom'].set_visible(False)
    fig_axes[2][i].spines['left'].set_visible(False)
    fig_axes[2][i].spines['right'].set_visible(False)
    
    fig_axes[1][i].set_xlim(-0.05, 0.05)  # Reduced x limits
    fig_axes[1][i].set_ylim(-1.5, 1.5)    # Reduced y limits
    fig_axes[1][i].spines['top'].set_visible(False)
    fig_axes[1][i].spines['bottom'].set_visible(False)
    fig_axes[1][i].spines['left'].set_visible(False)
    fig_axes[1][i].spines['right'].set_visible(False)
    draw_graph(fig_axes[1][i], on_table_probs[fidx])
    fig_axes[2][i].imshow(results["pngs"][i])


plt.tight_layout()

condorgmm_tex_dir = condorgmm.get_root_path() / "condorgmm_tex"
plt.savefig(condorgmm_tex_dir / "scene_graph_fig.pdf")
