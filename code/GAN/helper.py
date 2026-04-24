import itertools
import math
import time
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import GaussianBlur

def plot_cifar10_grid(training_data, test_data, sample_count=16, nrow=4, padding=2, figsize=(8, 8),
                      title='CIFAR-10 Images', title_style={'fontsize': 16, 'fontweight': 'bold', 'color': 'red'}):
    """
    Plots a grid of CIFAR-10 images with labels.

    Parameters:
    - training_data: Dataset containing the training set with attribute `classes`
                     for class names.
    - test_data: Iterable dataset for testing images and labels.
    - sample_count: Number of images to sample from test_data (default is 16).
    - nrow: Number of images per row in the grid (default is 4).
    - padding: Padding between images in the grid (default is 2).
    - figsize: Size of the matplotlib figure (default is (8, 8)).
    - title: Title for the plot (default is 'CIFAR-10 Images').
    - title_style: Dictionary of style parameters for the title text.

    Returns:
    - None. Displays the plotted grid.
    """
    # Get the class names from training data
    class_names = training_data.classes

    # Sample sample_count images and labels from test_data
    images, labels = zip(*tuple(itertools.islice(iter(test_data), sample_count)))
    images = torch.stack(images)
    
    # Create a grid of images
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=padding)
    grid = (grid + 1) / 2  # Denormalize if images are in [-1, 1]
    np_grid = grid.permute(1, 2, 0).numpy()
    
    # Create the figure and display the grid
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(np_grid)
    ax.set_title(title, **title_style)
    ax.axis('off')
    
    # Calculate grid rows (ensure proper handling if sample_count is not a perfect square)
    grid_cols = nrow
    grid_rows = math.ceil(sample_count / nrow)
    img_height = np_grid.shape[0] // grid_rows  
    img_width = np_grid.shape[1] // grid_cols 
    
    # Loop over each image to draw a rectangle and add the label
    for i, label in enumerate(labels):
        row = i // grid_cols
        col = i % grid_cols
        cell_x = col * img_width
        cell_y = row * img_height
        
        # Draw a dashed rectangle for the cell
        rect = patches.Rectangle((cell_x, cell_y), img_width, img_height, linewidth=2,
                                 edgecolor='black', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        
        # Center label text within the cell with a small vertical offset
        x_center = cell_x + img_width / 2
        y_center = cell_y + img_height / 2
        ax.text(x_center, y_center + img_height / 3, class_names[label],
                ha='center', va='center', fontsize=12, color='white',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=2))
    
    plt.show()

def check_network(net, input_shape, num_classes, max_params=10_000_000):
    """
    Validates a PyTorch network by checking:
    1. If it has trainable parameters.
    2. If the number of parameters is within a reasonable range.
    3. If the forward() method produces the correct output shape.
    4. If the output requires gradients.
    5. If the predict() method produces the correct output shape.

    Parameters:
    - net: The PyTorch neural network to validate.
    - input_shape: Tuple representing the input shape (default: (3, 32, 32)).
    - num_classes: Expected number of output classes (default: 10).
    - max_params: Maximum allowed number of parameters (default: 10M).

    Returns:
    - bool: True if all checks pass, False otherwise.
    """

    ok = True

    # Check if the network has parameters
    if len(tuple(net.parameters())) == 0:
        print('Your network does not define any parameters.')
        return False  # Early exit

    # Count trainable parameters
    param_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    if param_count > max_params:
        print(f'Warning: Your network consists of {param_count} parameters, '
              f'which is too much for this task!')

    # Check forward() and predict() methods with different batch sizes
    for batch_size in [1, 16, 32]:
        test_x = torch.zeros((batch_size, *input_shape), requires_grad=True)
        
        try:
            test_out = net(test_x)
            if test_out.shape != (batch_size, num_classes):
                print(f'The forward() method does not return the correct shape. '
                      f'Expected ({batch_size}, {num_classes}), but got {tuple(test_out.shape)}.')
                ok = False
            if not test_out.requires_grad:
                print('The output of forward() does not require a gradient.')
                ok = False
        except Exception as e:
            print(f'Error during forward() execution: {e}')
            return False

        try:
            test_pred = net.predict(test_x)
            if test_pred.shape != (batch_size,):
                print(f'The predict() method does not return the correct shape. '
                      f'Expected ({batch_size},), but got {tuple(test_pred.shape)}.')
                ok = False
        except Exception as e:
            print(f'Error during predict() execution: {e}')
            return False

    if ok:
        print('All checks passed! Move on!')

    return ok

def train_network(network, train_step, training_loader, test_loader, optimizer, epochs, baseline, device=None):
    """
    Trains a neural network and evaluates accuracy while tracking loss.

    Args:
        network (torch.nn.Module): The neural network model.
        training_loader (torch.utils.data.DataLoader): Dataloader for the training dataset.
        test_loader (torch.utils.data.DataLoader): Dataloader for the test dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        epochs (int): Number of training epochs.
        device (torch.device, optional): Compute device (CPU/GPU). Auto-detected if None.

    Returns:
        tuple: (loss_curve, accuracy_curve, epoch_markers) for plotting.
    """
    
    # Select device: GPU if available, else CPU.
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    print("=========================================")
    
    # Move model to the selected device.
    network.to(device)

    # Lists for tracking metrics.
    loss_curve = []
    accuracy_curve = []
    epoch_markers = []

    # Training loop over epochs.
    for epoch in range(epochs):
        network.train()
        losses = []
        batch_idx = 0
        time_start = time.time()

        # Iterate through training batches.
        for images, labels in training_loader:
            # Move data to device.
            images, labels = images.to(device), labels.to(device)

            # Forward and backward pass.
            loss = train_step(network, images, labels, optimizer)  # Calls our previous train_step function
            
            # Store loss and print periodically.
            losses.append(loss.detach().cpu())
            if batch_idx % 100 == 0:
                average_loss = sum(losses) / len(losses)
                loss_curve.append(average_loss)
                losses = []
                print(f"Epoch: {epoch+1:3d}/{epochs:3d}, Batch {batch_idx+1:5d}, Loss: {average_loss:.4f}")
            batch_idx += 1

        # Evaluate accuracy after each epoch.
        num_correct, num_total = 0, 0
        network.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                predicted_labels = network.predict(images)
                num_correct += (predicted_labels == labels).sum().item()
                num_total += images.shape[0]

        accuracy = (num_correct / num_total) * 100.0
        accuracy_curve.append(accuracy)
        epoch_markers.append(len(loss_curve) - 1)

        time_end = time.time()

        print(f"Correct: {num_correct}/{num_total}, Accuracy: ({accuracy:.3f}%)")
        print(f"Epoch took {time_end - time_start:.2f} seconds")
        print("=========================================")

    # Plot the training curves.
    plot_training_curves(loss_curve, accuracy_curve, epoch_markers, epochs, baseline)

    #return loss_curve, accuracy_curve, epoch_markers


def plot_training_curves(loss_curve, accuracy_curve, epoch_markers, epochs, baseline):
    """
    Plots loss and accuracy curves for training monitoring.

    Args:
        loss_curve (list): List of recorded loss values.
        accuracy_curve (list): List of recorded accuracy values.
        epoch_markers (list): Indices where epochs ended.
        epochs (int): Number of epochs.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle("Training Curves")

    # Loss curve
    ax1.set_title("Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_xticks(epoch_markers, tuple(range(1, epochs + 1)))
    ax1.plot(loss_curve, label="Loss", color="C0")
    ax1.scatter(epoch_markers, [loss_curve[i] for i in epoch_markers], color="C0")

    # Accuracy curve
    ax2.set_title("Accuracy on Test Dataset (%)")
    ax2.set_xlabel("Epoch")
    ax2.plot(range(1, epochs + 1), accuracy_curve, label="Accuracy", color="C1", marker="o")
    ax2.hlines(baseline, 1, epochs, linestyle="dashed", color="gray", alpha=0.5)

    fig.tight_layout()
    plt.show()

def test_cross_entropy_value(function):
    
    batch_size = 32
    num_classes = 10

    # Simulated logits and labels
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    labels = torch.randint(0, num_classes, size=(batch_size,), dtype=torch.long)

    # --- Compute losses ---

    loss_naive = function(logits, labels)
    loss_pytorch = torch.nn.functional.cross_entropy(logits, labels)

    # are the loss the same= alliw for a small tolerance
    equal = torch.allclose(loss_naive, loss_pytorch, atol=1e-5)
    if not equal:
        print("The values are not equal.")
        
    else:
        print("Nice. Loss is equal.")

    print(f"Your loss    : {loss_naive.item()}")
    print(f"Correct      : {loss_pytorch.item()}")

def visualize_linear1_weights_2d(network, input_shape, nrow=8, ncol=8, figsize=(8, 8)):
    """
    Visualizes the first linear layer weights, assuming the network processes 2D images.

    Args:
        network (torch.nn.Module): The neural network with a MLP whose first layer will be visualized.
        input_shape (tuple): Shape of the input to the linear layer (C, H, W).
        nrow (int): Number of filters per row in the grid.
        ncol (int): Number of filters per column in the grid.
        figsize (tuple): Size of the matplotlib figure.
    """
    # Find first linear layer
    linear = None
    for i, layer in enumerate(network.mlp):
        if isinstance(layer, torch.nn.Linear):
            linear = layer
            break
    if linear is None:
        raise ValueError("No linear layer found in the network.")

    # Extract and normalize weights
    W = linear.weight.cpu().detach().clone()
    W = W - W.min()
    W = W / W.max()
    W = W.reshape(-1, *input_shape)

    if W.shape[0] > nrow * ncol:
        print(f"Warning: Layer has {W.shape[0]} features, but only first {nrow * ncol} will be displayed.")

    W = W[:nrow * ncol]

    # Create grids for visualization
    grid = torchvision.utils.make_grid(W, nrow=nrow).permute(1, 2, 0)

    # Plot the results
    _, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(grid)
    ax.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_conv1_filters(network, nrow=8, blur_kernel=(3, 3), sigma=(0.5, 1.5), figsize=(10, 5)):
    """
    Visualizes the first convolutional layer filters before and after applying Gaussian blur.

    Args:
        network (torch.nn.Module): The neural network with a CNN whose first layer will be visualized.
        nrow (int): Number of filters per row in the grid.
        blur_kernel (tuple): Kernel size for Gaussian blur.
        sigma (tuple): Sigma range for Gaussian blur.
        figsize (tuple): Size of the matplotlib figure.
    """
    # Extract and normalize filters
    conv1 = network.cnn[0].cpu().weight.detach().clone()
    conv1 = conv1 - conv1.min()
    conv1 = conv1 / conv1.max()

    # Apply Gaussian blur to each filter
    gaussian_blur = GaussianBlur(kernel_size=blur_kernel, sigma=sigma)
    blurred_filters = torch.stack([gaussian_blur(filter.unsqueeze(0)).squeeze(0) for filter in conv1])

    # Create grids for visualization
    original_grid = torchvision.utils.make_grid(conv1, nrow=nrow).permute(1, 2, 0)
    blurred_grid = torchvision.utils.make_grid(blurred_filters, nrow=nrow).permute(1, 2, 0)

    # Plot the results
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(original_grid)
    axes[0].set_title('Original Filters - a bit noisy')
    axes[0].axis('off')

    axes[1].imshow(blurred_grid)
    axes[1].set_title('Blurred Filters - can you see edge detectors? :)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def total_variation_loss(img):
    """
    Computes total variation loss for smoothness regularization.
    """
    tv_loss = torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
              torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    return tv_loss

def generate_class_image(net, device, class_idx, iterations=1000, lr=0.001, tv_weight=0.000185, blur_every=50):
    """
    Generates an image that maximally activates the specified class in the network.

    Args:
        net (torch.nn.Module): The trained neural network.
        device (torch.device): Device to run the generation on.
        class_idx (int): Index of the class to visualize.
        iterations (int): Number of optimization steps.
        lr (float): Learning rate.
        tv_weight (float): Weight for total variation loss.
        blur_every (int): Frequency of applying Gaussian blur for regularization.

    Returns:
        torch.Tensor: Generated class image (normalized, shape: H x W x C).
    """
    img = torch.randn((1, 3, 32, 32), requires_grad=True, device=device)
    optimizer = torch.optim.AdamW([img], lr=lr)
    blur = GaussianBlur(kernel_size=3, sigma=1)

    for i in range(iterations):
        optimizer.zero_grad()

        # Apply blur every `blur_every` iterations
        if i % blur_every == 0:
            with torch.no_grad():
                img.data = blur(img.data)

        out = net(img)
        class_loss = -out[0, class_idx] + out[0, :].mean()
        tv_loss = total_variation_loss(img) * tv_weight
        loss = class_loss + tv_loss

        loss.backward()
        optimizer.step()

        img.data = img.data.clamp(-3, 3)

    # Normalize image for visualization
    img = img.detach().cpu()
    img = img - img.min()
    img = img / img.max()
    img = img.squeeze(0).permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    return img

def generate_images_for_all_classes(net, device, class_names, iterations=1050, lr=0.0015, show=True):
    """
    Generates and optionally displays class-activating images for each class in the model.

    Args:
        net (torch.nn.Module): Trained model.
        device (torch.device): CPU or GPU.
        class_names (list): List of class names corresponding to indices.
        iterations (int): Number of optimization iterations per class.
        lr (float): Learning rate.
        show (bool): Whether to display images with matplotlib.

    Returns:
        list: List of generated class images as torch.Tensor.
    """
    net.to(device)
    images = []

    for idx, class_name in enumerate(class_names):
        print(f'Generating image for class: {class_name}')
        img = generate_class_image(net, device, idx, iterations=iterations, lr=lr)
        images.append(img)

        if show:
            plt.figure(figsize=(2, 2))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'{class_name}')
            plt.show()

    return images

def test_linear_forward(MyLinear):
    # Set parameters for the test.
    batch_size = 4

    for in_features in [4, 8]:
        for out_features in [8, 16]:
            print(f"Testing with in_features={in_features}, out_features={out_features}")

            # Create random input tensors for the custom and native layers.
            # Clone the input so that both layers receive the same data.
            x = torch.randn(batch_size, in_features, requires_grad=True)
            x_native = x.clone().detach().requires_grad_()

            # Create the custom linear layer.
            custom_linear = MyLinear(in_features, out_features)

            # Create PyTorch's built-in Linear.
            native_linear = torch.nn.Linear(in_features, out_features)

            # Set the native_conv weights and bias to be identical to custom_conv's parameters.
            with torch.no_grad():
                native_linear.weight.copy_(custom_linear.weight)
                native_linear.bias.copy_(custom_linear.bias)

            # Forward pass.
            y_custom = custom_linear(x)
            y_native = native_linear(x_native)

            # print first 5 elements of the output

            assert torch.allclose(y_custom, y_native, atol=1e-3), "Forward pass outputs differ! Your linear: {}, PyTorch linear: {}".format(y_custom, y_native)

    print("Nice! Forward pass is correct. Move to the next task.")

def test_conv2d_forward(MyConv2d):

    # Set parameters for the test.
    batch_size = 2
    in_channels = 8
    out_channels = 2
    height, width = 4, 4

    for pad in [1,2]:
        for stride in [1,2]:
            for kernel_size in [3]:

                print(f"Testing with padding={pad}, stride={stride}, kernel_size={kernel_size}")

                # Create random input tensors for the custom and native layers.
                # Clone the input so that both convs receive the same data.
                x = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
                x_native = x.clone().detach().requires_grad_()

                # Create the custom convolution layer.
                custom_conv = MyConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)

                # Create PyTorch's built-in Conv2d.
                native_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)

                # Set the native_conv weights and bias to be identical to custom_conv's parameters.
                with torch.no_grad():
                    native_conv.weight.copy_(custom_conv.weight)
                    native_conv.bias.copy_(custom_conv.bias)

                # Forward pass.
                y_custom = custom_conv(x)
                y_native = native_conv(x_native)

                # print first 5 elements of the output

                assert torch.allclose(y_custom, y_native, atol=1e-3), "Forward pass outputs differ! Your conv2d: {}, PyTorch conv2d: {}".format(y_custom, y_native)

    print("Nice! Forward pass is correct. Move to the next task.")