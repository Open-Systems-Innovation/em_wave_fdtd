def vis(arrays):
    """
    Visualizes one or more NumPy arrays using Matplotlib.

    Parameters:
    arrays (numpy.ndarray or list of numpy.ndarray): 
        A single NumPy array or a list of arrays to visualize.
    """
    # Ensure arrays is a list of arrays
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]
    elif not isinstance(arrays, list) or not all(isinstance(a, np.ndarray) for a in arrays):
        raise ValueError("Input must be a NumPy array or a list of NumPy arrays.")

    num_plots = len(arrays)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))

    # Handle single subplot case
    if num_plots == 1:
        axes = [axes]

    for i, (array, ax) in enumerate(zip(arrays, axes)):
        ndim = array.ndim

        if ndim == 1:  # 1D array
            ax.plot(np.transpose(array), marker='o', linestyle='-', color='b')
            ax.set_title(f"1D Array {i + 1}")
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.grid(True)
        elif ndim == 2:  # 2D array
            im = ax.imshow(np.transpose(array), cmap='viridis', aspect='auto')
            fig.colorbar(im, ax=ax, label="Value")
            ax.set_title(f"2D Array {i + 1}")
            # Add cell borders
            ax.set_xticks(np.arange(-0.5, array.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, array.shape[0], 1), minor=True)
            ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
            ax.tick_params(which="minor", size=0)  # Hide minor ticks
        else:  # Higher dimensions
            ax.set_title(f"Unsupported Array {i + 1}")
            ax.text(0.5, 0.5, "Unsupported Array", 
                    ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.show()

