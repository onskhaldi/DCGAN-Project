import torch
import matplotlib.pyplot as plt
from sampling import create_gif
import imageio

def interpolate(generator, z1, z2, steps, dataset, interpolation = 'linear' ,plot_interpolation = True, save_as_gif = False):
    """
    Generates and optionally displays images from a z1 to a z2 with steps+1 images being created.

    Args:
        generator (torch.nn.Module): Trained generator.
        z1 (torch.tensor): Startin latent vector.
        z2 (torch.tensor): Ending latent vector.
        steps (int): Number of steps from z1 to z2.
        dataset (String): Name of the dataset.
        interpolation (String): Type of interpolation used (linear or spherical)
        plot_interpolation (bool): Whether to display images with matplotlib.

    Returns:
        list: List of generated images as torch.Tensor.
    """
    device = next(generator.parameters()).device
    generator.to('cpu')

    if(interpolation == 'linear'):
        m = torch.linspace(0,1,steps)[:,None]
        path = (1 - m)* z1 + m*z2
        path = path.permute(2,1,0,3)
    elif(interpolation == 'spherical'):
        z1 = torch.squeeze(z1)
        z2 = torch.squeeze(z2)
        m = torch.linspace(0,1,steps)
        results = [slerp(m[i],z1,z2) for i in range(steps)] #apply spherical interpolation
        path = torch.stack(results)  # shape: (n, m)
        path = torch.unsqueeze(path,2)
        path = torch.unsqueeze(path,2)
    else:
        print("Interpolation method does not exist")
        return

    #generate images
    print(path.shape)
    images = generator(path)
    #reshape images according to dataset
    if(dataset == 'celebA'):
        images = torch.reshape(images,(steps,3,64,64)) # reshape for testing
    elif(dataset == 'mnist'):
        images = torch.reshape(images,(steps,1,32,32))
    elif(dataset == 'cifar10'):
        images = torch.reshape(images,(steps,3,32,32))


    # display images
    if(plot_interpolation):
        print('plotting images')
        image_scale = 2

        #calculate plot shape
        if(steps < 5):
            cols = steps
            rows = 1
        else:
            cols = 5
            rows = (steps // cols)

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
    if(save_as_gif):
        title = "Interpolation_"+dataset+"_"+interpolation+"_"+str(steps)+"_steps"
        create_gif(images,dataset,title)
    return images


# spherical linear interpolation (slerp)
def slerp(val, low, high):
	omega = torch.arccos(torch.clip(torch.dot(low/torch.norm(low), high/torch.norm(high)), -1, 1))
	so = torch.sin(omega)
	if so == 0:
		# L'Hopital's rule/LERP
		return (1.0-val) * low + val * high
	return torch.sin((1.0-val)*omega) / so * low + torch.sin(val*omega) / so * high