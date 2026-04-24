import matplotlib.pyplot as plt
import torch



def visualize_loss(generator_loss,discriminator_loss,accuracy,epoch_markers,epochs,baseline=0):
    """
    Plots loss of generator and discriminator and accuracy curves for training monitoring.

    Args:
        generator_loss (list): List of recorded loss values for the generator.
        discriminator_loss (list): List of recorded loss values for the discriminator.
        accuracy (list): List of recorded accuracy values.
        epoch_markers (list): Indices where epochs ended.
        epochs (int): Number of epochs.
        baseline (int): Baseline value for accuracy plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Training Curves")

    # Loss curves
    ax1.set_title("Loss curves")
    ax1.set_xlabel("Epoch")
    ax1.set_xticks(epoch_markers, tuple(range(1, epochs + 1)))
    ax1.plot(discriminator_loss, label="Loss Discriminator", color="#FF7F27")
    ax1.scatter(epoch_markers, [discriminator_loss[i] for i in epoch_markers], color="#FF7F27")
    ax1.plot(generator_loss, label="Loss Generator", color="#0BA938")
    ax1.scatter(epoch_markers, [generator_loss[i] for i in epoch_markers], color="#0BA938")
    # Accuracy curve
    ax2.set_title("Accuracy on Test Dataset (%)")
    ax2.set_xlabel("Epoch")
    ax2.plot(range(1, epochs + 1), accuracy, label="Accuracy", color="C1", marker="o")
    ax2.hlines(baseline, 1, epochs, linestyle="dashed", color="gray", alpha=0.5)

    fig.tight_layout()
    ax1.legend()
    plt.show()

def generate_sampels_mnist(generator, device, latent_dimensions, number_of_images,image_scale=2, show=True):
    """
    Generates and optionally displays images for the mnist dataset.

    Args:
        generator (torch.nn.Module): Trained generator.
        device (torch.device): CPU or GPU.
        latent_dimensions (int): Number of dimensions for latent vector
        number_of_images (int): Number of images to be generated.
        image_scale (int): Size of images in generated plot.
        show (bool): Whether to display images with matplotlib.

    Returns:
        list: List of generated images as torch.Tensor.
    """
    generator.to(device)
    z = torch.randn(number_of_images, latent_dimensions,1,1,device = device) # latent noise
    images = generator(z) # sample images
    images = torch.reshape(images,(number_of_images,1,32,32)) # reshape for testing

    # define format of plot
    if(number_of_images < 5):
        cols = number_of_images
        rows = 1
    else:
        cols = 5
        rows = (number_of_images // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*image_scale, rows*image_scale))
    axes = axes.flatten()

    # display each image in gray scale
    for i, ax in enumerate(axes):
        if i < len(images):
            if(device == 'cpu'):
                ax.imshow(images[i][0].detach().numpy(), cmap='gray')
            else: 
                ax.imshow(images[i][0].cpu().detach().numpy(), cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')
        
    plt.tight_layout()
    plt.show()

    return images

def generate_sampels_cifar10(generator, device, latent_dimensions, number_of_images, image_scale=2, show=True):
    """
    Generates and optionally displays images for the cifar10 dataset.

    Args:
        generator (torch.nn.Module): Trained generator.
        device (torch.device): CPU or GPU.
        latent_dimensions (int): Number of dimensions for latent vector
        number_of_images (int): Number of images to be generated.
        image_scale (int): Size of images in generated plot.
        show (bool): Whether to display images with matplotlib.

    Returns:
        list: List of generated images as torch.Tensor.
    """
    generator.to(device)
    z = torch.randn(number_of_images, latent_dimensions,1,1,device = device) # latent noise
    images = generator(z) # sample images
    images = torch.reshape(images,(number_of_images,3,32,32)) # reshape for testing

    # define format of plot
    if(number_of_images < 5):
        cols = number_of_images
        rows = 1
    else:
        cols = 5
        rows = (number_of_images // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*image_scale, rows*image_scale))
    axes = axes.flatten()

    # display each image
    for i, ax in enumerate(axes):
        if i < len(images):
            if(device == 'cpu'):
                img = images[i].detach().permute(1, 2, 0).numpy()  # change dimensions from [C, H, W] to [H, W, C]
            else:                 
                img = images[i].cpu().detach().permute(1, 2, 0).numpy()  # change dimensions from [C, H, W] to [H, W, C]
            img = (img + 1) / 2  # normalize to range [0...1]
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.axis('off')
        
    plt.tight_layout()
    plt.show()

    return images

def generate_sampels_celebA(generator, device, latent_dimensions, number_of_images, image_scale = 2,show=True):
    """
    Generates and optionally displays images for the celebA dataset.

    Args:
        generator (torch.nn.Module): Trained generator.
        device (torch.device): CPU or GPU.
        latent_dimensions (int): Number of dimensions for latent vector
        number_of_images (int): Number of images to be generated.
        image_scale (int): Size of images in generated plot.
        show (bool): Whether to display images with matplotlib.

    Returns:
        list: List of generated images as torch.Tensor.
    """
    generator.to(device)
    z = torch.randn(number_of_images, latent_dimensions,1,1,device= device) # latent noise
    images = generator(z) # sample images
    images = torch.reshape(images,(number_of_images,3,64,64)) # reshape for testing

    # define format of plot
    if(number_of_images < 5):
        cols = number_of_images
        rows = 1
    else:
        cols = 5
        rows = (number_of_images // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*image_scale*2, rows*image_scale*2))
    axes = axes.flatten()

    # display each image
    for i, ax in enumerate(axes):
        if i < len(images):
            if(device == 'cpu'):
                img = images[i].detach().permute(1, 2, 0).numpy()  # change dimensions from [C, H, W] to [H, W, C]
            else:                 
                img = images[i].cpu().detach().permute(1, 2, 0).numpy()  # change dimensions from [C, H, W] to [H, W, C]    
                img = (img + 1) / 2  # normalize to range [0...1]
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.axis('off')
        
    plt.tight_layout()
    plt.show()

    return images

def save_plot(images,filename="placeholder",image_scale=2, device ='cpu'):
    """
    Stores a tensor of images as an image under "outputs/generated/plots/dataset/filename.png

    Args:
        images (list): Tensor that contains the image data for a figure
        filename (String): Name of the Image that will be saved (no .png) 
        nimage_scale (int): Size of the plots in the resulting image
    """
    print(images.shape)
    number_of_images = images.shape[0]
    channels = images.shape[1]
    size = images.shape[2]
    dataset = "cifar10"
    if(size == 64):
        dataset = "celebA"
        image_scale *= 2
    if(number_of_images < 5):
        cols = number_of_images
        rows = 1
    else:
        cols = 5
        rows = (number_of_images // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*image_scale, rows*image_scale))
    axes = axes.flatten()

    # Display each image
    if(channels == 3):
        for i, ax in enumerate(axes):
            if i < len(images):
                if(device != 'cpu'):
                    img = images[i].cpu().detach().permute(1, 2, 0).numpy()
                else:
                    img = images[i].detach().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())  # Normalize to range [0...1]
                ax.imshow(img)
                ax.axis('off')
            else:
                ax.axis('off')
    else: 
        dataset = "mnist"
        for i, ax in enumerate(axes):
            if i < len(images):
                if(device != 'cpu'):
                    ax.imshow(images[i][0].cpu().detach().numpy(), cmap='gray')
                else:
                    ax.imshow(images[i][0].detach().numpy(), cmap='gray')
                ax.axis('off')
            else:
                ax.axis('off')

        
    plt.tight_layout()
    plt.savefig("outputs/generated/plots/"+dataset+"/"+filename +".png")
    print('File saved as "outputs/generated/plots/'+dataset+"/"+filename +'.png"')

def visualize_latent_space(generator, device, latent_vector, channels, size, show = False):
    generator.to(device)
    images = generator(latent_vector) # sample images
    number_of_images = latent_vector.shape[0]
    images = torch.reshape(images,(number_of_images,channels,size,size)) # reshape for testing

    # define format of plot
    if(show):
        if(number_of_images < 5):
            cols = number_of_images
            rows = 1
        else:
            cols = 5
            rows = (number_of_images // cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        axes = axes.flatten()

        # display each image
        for i, ax in enumerate(axes):
            if i < len(images):
                img = images[i].detach().permute(1, 2, 0).numpy()  # change dimensions from [C, H, W] to [H, W, C]
                img = (img + 1) / 2 # normalize to range [0...1]
                ax.imshow(img)
                ax.axis('off')
            else:
                ax.axis('off')
            
        plt.tight_layout()
        plt.show()
    return images