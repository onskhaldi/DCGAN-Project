import torch
import torch.nn as nn
import math

# Initilize weight for conv or batchNorm
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator-Network for DCGAN

class DCGANGenerator(nn.Module):
    # Input = Tensor mit Shape (BachtSize, Latent_dim, 1, 1)
    # Output = (BatchSize, Channel , Size, Size)

    def __init__(self, 
                 latent_dim, 
                 img_channels, 
                 feature_maps, 
                 image_size
                 ):

        """
        Dynamic DCGAN Generator for 32x32 or 64x64 images.
        Args:
            latent_dim (int): Size of the input noise vector 
            img_channels (int): Number of channels in the output image: 1 for MNIST, 3 for CIFAR-10 and CelebA.
            feature_maps (int): Base number of feature maps.
            image_size (int): Target image size ( 32 or 64 )
        """
        super(DCGANGenerator, self).__init__()

        #Extract the number of layers
        n_layers = int(math.log2(image_size // 4)) + 1

        # To save Layers
        layers = []

        # Start: 1x1 → 4x4
        layers.append( 
            # Using fractional-strided convolutions (upsampling)
            nn.ConvTranspose2d(
                in_channels=latent_dim, 
                out_channels=feature_maps * 8, # (64-->512)
                kernel_size=4, 
                stride=1, 
                padding=0, 
                bias=False
                )
            )
        layers.append(
            # Applies batch normalization to stabilize training 
            nn.BatchNorm2d(feature_maps * 8))
        layers.append(
            #ActivationFunktion max(0,x)
            # In-place ReLU saves RAM
            nn.ReLU(inplace=True))


        # Starting channel size for next layer
        in_channels = feature_maps * 8

        # From Step 2 to n: Add upsampling layers dynamically based on n_layers
        for i in range (n_layers - 2):

            out_channels = in_channels // 2
            layers.append(
                nn.ConvTranspose2d(in_channels, 
                                   out_channels, 
                                   kernel_size=4, 
                                   stride=2, 
                                   padding=1, 
                                   bias=False
                                   )
                        )
            layers.append(
                # Normalize activations
                nn.BatchNorm2d(out_channels))
            layers.append(
                nn.ReLU(inplace=True))
            
            # Update for next layer
            in_channels = out_channels

        # Final Layer : Output Image
        layers.append(
            nn.ConvTranspose2d(in_channels, 
                               img_channels, # 3 OR  1
                               kernel_size=4, 
                               stride=2, 
                               padding=1, 
                               bias=False
                               )
                    )
        
        # Use Tanh() at output 
        layers.append(nn.Tanh())

        # Wrap all layers into a Sequential module
        self.gen_network = nn.Sequential(*layers)

        """
         Input: latent vector (B, latent_dim, 1, 1)
         Output: generated image (B, img_channels, image_size, image_size)
        """
    def forward(self, x):
        # Defines the forward pass through the generator network.
        return self.gen_network(x)


        
# Discriminator-Network for DCGAN

# sizes that are not supported in this discriminator
#     Sizes not powers of 2 
#     Sizes smaller than 32x32
#     Very large sizes (256x256 or higher) may cause high GPU memory usage 
#

class DCGANDiscriminator(nn.Module):

    def __init__(self, 
                 img_channels, 
                 feature_maps, 
                 image_size
                 ):
        
        """
        Dynamic DCGAN Discriminator for 32x32 , 64x64 or 128*128 images.
        Args:
            img_channels (int): Number of channels in the input image.
            feature_maps (int): Base number of feature maps.
            image_size (int): Image resolution (32 or 64).
        """

        super(DCGANDiscriminator, self).__init__()

        # Compute how many downsampling layers we need to reach 4x4
        n_layers = int(math.log2(image_size)) - 2  

        # to save the Layers 
        layers = []

         # Initial layer: no BatchNorm, just Conv + LeakyReLU
        layers.append(
            #Conv for Downsampling
            nn.Conv2d(img_channels, 
                      feature_maps,
                        kernel_size=4, 
                        stride=2, 
                        padding=1, 
                        bias=False
                    )
        )
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        in_channels = feature_maps

        # Add intermediate layers
        for i in range(n_layers - 1):

            # double channels 
            out_channels = in_channels * 2
            layers.append(
                nn.Conv2d(in_channels, 
                          out_channels, 
                          kernel_size=4, 
                          stride=2, 
                          padding=1, 
                          bias=False
                          )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            in_channels = out_channels

        # Final layer: Output 1×1 (single value)
        layers.append(
            nn.Conv2d(in_channels, 
                      out_channels=1, 
                      kernel_size=4, 
                      stride=1, 
                      padding=0, 
                      bias=False
                      )
        )
        # no ACtivation (BCEWithLogitsLoss) have a sigmoid 

        self.disc_network = nn.Sequential(*layers)

    def forward(self, x):
        output = self.disc_network(x)
        return torch.flatten(output, 1)  # Shape of the output Vector : (B, 1)


