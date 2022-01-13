from im2scene.discriminator import conv, patch_discriminator


discriminator_dict = {
    'dc': conv.DCDiscriminator,
    'resnet': conv.DiscriminatorResnet,
}


patch_discriminator_dict = {
    'patch': patch_discriminator.StyleGAN2PatchDiscriminator
}