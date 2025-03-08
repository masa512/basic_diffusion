


import torch.functional as F 
import torch.nn as nn
######## Pure Unet Implementation ###############

class cbr_block(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size,include_relu = True ,include_bn = True):

        super().__init__()

        # Initialize CBR Block with just conv
        self.cbr = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride = 1,
                padding = (kernel_size-1)//2
            )
        )

        # Add relu if needed
        if include_relu:
            self.cbr.append(
                nn.ReLU()
            )

        # Add bn if needed
        if include_bn:
            self.cbr.append(
                nn.BatchNorm2d(num_features=out_channels)
            )

    def forward(self,x):

        return self.cbr(x)

class double_cbr_block(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size,include_relu = True ,include_bn = True):

        super().__init__()

        # initialize the two cbr blocks
        self.cbr1 = cbr_block(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size = kernel_size,
            include_relu= include_relu,
            include_bn = include_bn
        )

        self.cbr2 = cbr_block(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size = kernel_size,
            include_relu= include_relu,
            include_bn = include_bn
        )
    
    def forward(self,x):

        x = self.cbr1(x)
        x = self.cbr2(x)
        return x

class encoder(nn.Module):
    
    def __init__(self,input_channels,in_channels, kernel_size, depth = 1 ,include_relu = True ,include_bn = True):

        super().__init__()

        # Depth excludes the input block

        # Define input block
        self.input_block = double_cbr_block(
                in_channels= input_channels,
                out_channels= in_channels,
                kernel_size = kernel_size,
                include_relu= include_relu,
                include_bn = include_bn
        )
        
        self.enc_seq = nn.ModuleList()

        for i in range(1,depth+1):

            self.enc_seq.add_module(f'enc{i}',
                double_cbr_block(
                in_channels= in_channels * (2**(i-1)),
                out_channels= in_channels * (2**(i)),
                kernel_size = kernel_size,
                include_relu= include_relu,
                include_bn = include_bn
            )
            )
        
        self.pool = nn.MaxPool2d(2,2)
    
    def forward(self,x):

        # Input
        x = self.input_block(x)

        # Residual outputs
        res = {}

        # each encoder block
        for i, e in enumerate(self.enc_seq):
            # First apply the conv layer
            print(i,x.size(),e)
            r = e(x)
            res[i+1] = r
            # Apply pool
            x = self.pool(x)
        
        return x,res

