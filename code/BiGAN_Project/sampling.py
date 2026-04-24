import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import imageio
import os


def show_reconstructions(encoder, generator, dataloader, device=None, num_images=8):
    encoder.eval()
    generator.eval()
    
    if device is None:
        device = next(generator.parameters()).device
    
    with torch.no_grad():
        # Lade einen Batch
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            
            # Encode und dekodiere die Bilder
            z = encoder(imgs)
            recon_imgs = generator(z)
            
            # Zeige Originale und Rekonstruktionen nebeneinander
            imgs = imgs.cpu()
            recon_imgs = recon_imgs.cpu()
            
            plt.figure(figsize=(num_images*2, 4))
            
            for i in range(num_images):
                # Original
                plt.subplot(2, num_images, i+1)
                if imgs.shape[1] == 1:
                    plt.imshow(imgs[i, 0], cmap='gray')
                else:
                    img = imgs[i].permute(1, 2, 0).numpy()
                    img = (img + 1) / 2 
                    plt.imshow(img)
                plt.axis('off')
                if i == 0:
                    plt.title('Original')
                    
                #Reconstruction
                plt.subplot(2, num_images, i + 1 + num_images)
                if recon_imgs.shape[1] == 1:
                    recon = recon_imgs[i, 0].numpy()
                    plt.imshow(recon, cmap='gray')
                else:
                    recon = recon_imgs[i].permute(1, 2, 0).numpy()
                    recon = (recon + 1) / 2  
                    plt.imshow(recon)
                plt.axis('off')
                if i == 0:
                    plt.title('Reconstruction')
            
            plt.tight_layout()
            plt.show()
            break


def visualize_results(results):
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
    (losses_D,
    losses_G,
    losses_E,
    test_D,
    test_G,
    test_E,
    learning_rates,
    d_real_losses,
    d_fake_losses,
    g_losses_per_batch,
    e_losses_per_batch) = results

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].plot(losses_D, label="Train D Loss")
    axs[0].plot(losses_G, label="Train G Loss")
    axs[0].plot(losses_E, label="Train E Loss")
    axs[0].scatter(range(len(test_D)), test_D, label="Test D Loss", color='orange')
    axs[0].scatter(range(len(test_G)), test_G, label="Test G Loss", color='green')
    axs[0].scatter(range(len(test_E)), test_E, label="Test E Loss", color='blue')
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].set_title("Training & Test Losses")
    axs[0].grid()

    axs[1].plot(learning_rates, label="Learning Rate")
    axs[1].set_xlabel("Logged Batch (every 150th)")
    axs[1].set_ylabel("Learning Rate")
    axs[1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2e'))
    axs[1].set_title("Learning Rate Over Time")
    axs[1].grid()
    axs[1].legend()

    axs[2].plot(d_real_losses, label="D Real Loss")
    axs[2].plot(d_fake_losses, label="D Fake Loss")
    axs[2].plot(g_losses_per_batch, label="G Loss")
    axs[2].plot(e_losses_per_batch, label="E Loss")
    axs[2].set_xlabel("Logged Batch (every 150th)")
    axs[2].set_ylabel("Loss")
    axs[2].set_title("Loss Components")
    axs[2].grid()
    axs[2].legend()

    plt.tight_layout()
    plt.show()

def sample(generator, dataset, latent_dimensions, number_of_images, image_scale = 2, show=True):
    """
    Generates and optionally displays images for the given generator and dataset.

    Args:
        generator (torch.nn.Module): Trained generator.
        dataset (String): Name of dataset.
        latent_dimensions (int): Number of dimensions for latent vector
        number_of_images (int): Number of images to be generated.
        image_scale (int): Size of images in generated plot.
        show (bool): Whether to display images with matplotlib.

    Returns:
        list: List of generated images as torch.Tensor.
    """
    device = next(generator.parameters()).device
    generator.to('cpu')
    z = torch.randn(number_of_images, latent_dimensions,1,1) # latent noise
    images = generator(z) # sample images
    if(dataset == 'celebA'):
        images = torch.reshape(images,(number_of_images,3,64,64)) # reshape for testing
    elif(dataset == 'mnist'):
        images = torch.reshape(images,(number_of_images,1,32,32))
    elif(dataset == 'cifar10'):
        images = torch.reshape(images,(number_of_images,3,32,32))
    else:
        print("No such dataset")
        return
    #plot images 
    if(show):
        #calculate plot shape
        if(number_of_images < 5):
            cols = number_of_images
            rows = 1
        else:
            cols = 5
            rows = (number_of_images // cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols*image_scale, rows*image_scale))
        axes = axes.flatten()
        #plot images
        if(dataset == 'mnist'):
            for i, ax in enumerate(axes):
                if i < len(images):
                    ax.imshow(images[i][0].cpu().detach().numpy(), cmap='gray')
                    ax.axis('off')
                else:
                    ax.axis('off')
        if(dataset == 'cifar10' or dataset == 'celebA'):
            for i, ax in enumerate(axes):
                if i < len(images):
                    img = images[i].cpu().detach().permute(1, 2, 0).numpy()
                    img = (img + 1) / 2  # Normalize to range [0...1]
                    ax.imshow(img)
                    ax.axis('off')
                else:
                    ax.axis('off')
        plt.tight_layout()
        plt.show()
    generator.to(device)
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
                img = (img + 1) / 2  # Normalize to range [0...1]
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

    #filename = f"{dataset}_{filename}.png"
    #plt.tight_layout()
    #plt.savefig(filename)
    #print(f"File saved as {filename}")

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
                img = (img+1) / 2 # normalize to range [0...1]
                ax.imshow(img)
                ax.axis('off')
            else:
                ax.axis('off')
            
        plt.tight_layout()
        plt.show()
    return images

def create_gif(images, dataset, title):
    #----------------------------------
    output_dir = os.path.join("outputs", "interpolations", dataset)
    os.makedirs(output_dir, exist_ok=True)
    #------------------
    os.makedirs("frames", exist_ok=True)

    #Generate frames
    filenames = []
    for i in range(images.shape[0]):
        plt.figure(figsize=(2, 2))
        if(dataset == 'mnist'):
            plt.imshow(images[i][0].cpu().detach().numpy(), cmap='gray')
            plt.axis('off')
             
        if(dataset == 'cifar10' or dataset == 'celebA'):
            img = images[i].cpu().detach().permute(1, 2, 0).numpy()
            img = (img + 1) / 2  # Normalize to range [0...1]
            plt.imshow(img)
            plt.axis('off')
        
        filename = f"frames/frame_{i:03d}.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        filenames.append(filename)

    #Create a GIF
    gif_path = os.path.join(output_dir, f"{title}.gif") #new

    with imageio.get_writer(title+".gif", mode='I', duration=0.1) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    #Clean up frames
    for filename in filenames:
        os.remove(filename)
