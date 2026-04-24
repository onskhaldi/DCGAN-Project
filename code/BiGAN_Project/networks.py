import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Initilize weight for conv or batchNorm
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
# Generator-Network for BIGAN
# Input: latent vector (B, latent_dim, 1, 1)
# Output : (B, img_channels, image_size, image_size)

class BiGANGenerator(nn.Module):

    def __init__(self, 
                 latent_dim, 
                 img_channels, 
                 feature_maps, 
                 image_size
                 ):

        """
        Dynamic BiGAN Generator for 32x32 or 64x64 images.
        Args:
            latent_dim (int): Size of the input noise vector 
            img_channels (int): Number of channels in the output image: 1 for MNIST, 3 for CIFAR-10 
            feature_maps (int): Base number of feature maps.
            image_size (int): Target image size ( 32 or 64 )
        """
        super(BiGANGenerator, self).__init__()

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
                               )
                    )
        
        # Use Tanh() at output 
        layers.append(nn.Tanh())

        # Wrap all layers into a Sequential module
        self.gen_network = nn.Sequential(*layers)
        self.apply(init_weights)

        """
         Input: latent vector (B, latent_dim, 1, 1)
         Output: generated image (B, img_channels, image_size, image_size)
        """
    def forward(self, x):
        # Defines the forward pass through the generator network.
        return self.gen_network(x)

# Discriminator-Network for BIGAN
# Input: latent vector (B, latent_dim, 1, 1)
# Input : Bild (B, C, H, W)
# Output : (B,1)

class BiGANDiscriminator(nn.Module):
    def __init__(
            self, 
            img_channels, 
            latent_dim,  
            feature_maps,
            image_size
            ):
        
        """
        Discriminator: Takes image x and latent vector z, predicts if the pair is real or fake.
        Two separate paths for x and z, followed by feature fusion.
        """

        super(BiGANDiscriminator, self).__init__()

        # Compute how many downsampling layers
        n_layers = int(math.log2(image_size)) - 2  

        # to save the Layers 
        layers = []

        # Initial layer: no BatchNorm, just Conv + LeakyReLU
        layers.append(nn.Conv2d(
            img_channels, 
            feature_maps, 
            kernel_size=4, 
            stride=2, 
            padding=1))
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        in_channels = feature_maps
        # Add intermediate layers
        for i in range(1, n_layers):

            # double channels 
            out_channels = in_channels * 2

            layers.append(nn.Conv2d( in_channels,
                                     out_channels, 
                                     kernel_size=4, 
                                     stride=2, 
                                     padding=1))

            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels

        self.img_net = nn.Sequential(*layers)


        # Latenter Vektor z → Feature Map (z_feat)
        self.z_net = nn.Sequential(
            #upsampling
            nn.ConvTranspose2d(latent_dim, 
                               in_channels * 4,                            
                               kernel_size=4, 
                               stride=1, 
                               padding=0, 
                              ),  # 1x1 → 4x4

            nn.BatchNorm2d(in_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels * 4, 
                      in_channels * 2, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),   # 4x4 → 4x4

            nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels * 2, 
                      in_channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),          # 4x4 → 4x4

            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True)

        )

        # Final classifier network
        self.final_net = nn.Sequential(

            # downsampling 
            nn.Conv2d(in_channels * 2, 
                      in_channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1
                      ),  # Fusion von x & z
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels, 1, 
                      kernel_size=4, 
                      stride=1, 
                      padding=0)  # → Output: 1×1
        )
        self.apply(init_weights)

    def forward(self, x, z):

        x_feat = self.img_net(x)  # (B, C, 4, 4)
        z_feat = self.z_net(z)    # (B, C, 4, 4)

        # Fusion 
        combined = torch.cat([x_feat, z_feat], dim=1)  # (B, 2C, 4, 4)
        out = self.final_net(combined)                 # (B, 1, 1, 1)

        # For BCEWithLogitsLoss
        return torch.flatten(out, 1)  # (B, 1)

#Encoder-Network for BiGAN
# Input :  (B, img_channels, image_size, image_size)
# Output : (B, latent_dim, 1, 1)
class BiGANEncoder(nn.Module):

    def __init__(self, 
                 img_channels, 
                 latent_dim, 
                 feature_maps, 
                 image_size):
        
        """
        Encoder: Maps image x to latent vector z.
        Reverse of the generator: uses downsampling via convolution.
        Compatible with 32x32, 64x64, 128x128

        """
        
        super(BiGANEncoder, self).__init__()

        n_layers = int(math.log2(image_size)) - 2

        layers = []

        # First convolution layer (no BatchNorm)
        layers.append(nn.Conv2d(
                                in_channels=img_channels, 
                                out_channels= feature_maps, 
                                kernel_size=4, 
                                stride=2, 
                                padding=1
                               )
                     )
        
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        in_channels = feature_maps

        # Downsampling layers
        for i in range(n_layers - 1):

            out_channels = in_channels * 2
            layers.append(nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=4, 
                stride=2, 
                padding=1 ))
            
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            in_channels = out_channels


        # Final layer: output latent vector z
        layers.append(nn.Conv2d(in_channels=in_channels, 
                                out_channels=latent_dim, 
                                kernel_size=4, 
                                stride=1, 
                                padding=0
                                )) # Output: (B, latent_dim, 1, 1)

        self.encoder_network = nn.Sequential(*layers)
        self.apply(init_weights) # for stability

    def forward(self, x):
        z = self.encoder_network(x)   # Shape: (B, latent_dim, 1, 1)
        #z = F.normalize(z, dim=1)  # Normalize vector length
        return z    


