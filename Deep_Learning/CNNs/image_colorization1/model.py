import torch.nn as nn
import torch

class DownConv(nn.Module):
    def __init__(
            self, kernel, in_channels, out_channels , padding , residual = False
    ):
        super(DownConv, self).__init__()
        
        # properties of class
        self.residual = residual
        self.conv1 = nn.Conv2d(in_channels ,  out_channels , kernel_size = kernel , padding = padding)
        self.conv1_2 = nn.Conv2d(out_channels , out_channels, kernel_size = kernel , padding = padding)
        self.maxpool = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels , out_channels, kernel_size = kernel , padding = padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################
        if self.residual == False:
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.bn1(x)
            x = self.relu1(x)
            
        else:
            residual_input = self.conv1(x)
            #print("residual" , residual_input.shape)
            x = self.conv1_2(residual_input)
            x = self.bn1(x)
            x = self.relu1(x)
            #print("first module", x.shape)
            
            
            x = x + residual_input
            
            x = self.conv2(x)
            x = self.maxpool(x)
            x = self.bn2(x)
            x = self.relu2(x)
            
            #print("second module:" , x.shape)

        return x
        


class UpConv(nn.Module):
    def __init__(
            self, kernel, in_channels, out_channels , padding , residual = False,
    ):
        super(UpConv, self).__init__()

        # properties of class
        self.residual = residual
        self.conv1 = nn.Conv2d(in_channels , out_channels, kernel_size = kernel , padding = padding)
        self.conv1_2 = nn.Conv2d(out_channels , out_channels, kernel_size = kernel , padding = padding)

        self.upsample = nn.Upsample(scale_factor = 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels , out_channels, kernel_size = kernel , padding = padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################

        
        if self.residual == False:
            x = self.conv1(x)
            x = self.upsample(x)
            x = self.bn1(x)
            x = self.relu1(x)
            
        else:
            residual_input = self.conv1(x)
            #print("residual" , residual_input.shape)
            x = self.conv1_2(residual_input)
            x = self.bn1(x)
            x = self.relu1(x)
            #print("first module", x.shape)
            
            
            x = x + residual_input
            
            x = self.conv2(x)
            x = self.upsample(x)
            x = self.bn2(x)
            x = self.relu2(x)
            
            #print("second module:" , x.shape)
        
        return x


class Bottleneck(nn.Module):
    def __init__(
            self, kernel, in_channels, out_channels , padding , residual = False
    ):
        super(Bottleneck, self).__init__()

        # properties of class
        self.residual = residual

        self.conv1 = nn.Conv2d(in_channels , out_channels, kernel_size = kernel , padding=padding)
        self.conv1_2 = nn.Conv2d(out_channels , out_channels, kernel_size = kernel , padding = padding)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels , out_channels, kernel_size = kernel , padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################
        if self.residual == False:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
        else:
            residual_input = self.conv1(x)
            #print("residual" , residual_input.shape)
            x = self.conv1_2(residual_input)
            x = self.bn1(x)
            x = self.relu1(x)
            #print("first module", x.shape)
            
            
            x = x + residual_input
            
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            
            #print("second module:" , x.shape)

        return x

class BaseModel(nn.Module):
    def __init__(
            self, kernel, num_filters, num_colors, in_channels=1, padding=1
    ):
        super(BaseModel, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        # Other properties if needed
        
        self.final_conv = nn.Conv2d(in_channels = num_colors , out_channels = num_colors , kernel_size = kernel , padding = 1)
        
        # Down part of the model, bottleneck, Up part of the model, final conv
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################
        self.down1 = DownConv(kernel =kernel, in_channels = in_channels, out_channels = num_filters , padding=1 )
        self.down2 = DownConv(kernel = kernel, in_channels = num_filters, out_channels = num_filters *2  , padding=1)
        self.bottleneck = Bottleneck(kernel = kernel, in_channels = num_filters*2, out_channels = num_filters *2 , padding = 1)
        self.up1 = UpConv(kernel = kernel, in_channels = num_filters*2, out_channels = num_filters , padding=1)
        self.up2 = UpConv(kernel = kernel, in_channels = num_filters, out_channels = num_colors, padding=1)

        
        
    def forward(self, x):
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################
        #print("in" , x.shape)
        down1_out = self.down1(x)
        #print("down1:",down1_out.shape)

        down2_out = self.down2(down1_out)
        #print("down2:",down2_out.shape)

        bottleneck_out = self.bottleneck(down2_out)
        #print("bottel:", bottleneck_out.shape)

        up1_out = self.up1(bottleneck_out)
        #print("up1:" , up1_out.shape)

        up2_out = self.up2(up1_out)
        #print("up2:" ,up2_out.shape)
        
        final_output = self.final_conv(up2_out)
        #print("final:" , final_output.shape)

        return final_output


class CustomUNET(nn.Module):
    def __init__(
            self, kernel , num_filters, num_colors, in_channels=1, out_channels=3
    ):
        super(CustomUNET, self).__init__()


        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        # Other properties if needed
        
        self.final_conv = nn.Conv2d(in_channels = num_colors , out_channels = num_colors , kernel_size = kernel , padding = 1)
        self.skip_conv1 = nn.Conv2d(256 , 128 , 1)
        self.skip_conv2 = nn.Conv2d(128 , 64 , 1)
        self.skip_conv3 = nn.Conv2d(25 , 24 , 1)

        # Down part of the model, bottleneck, Up part of the model, final conv
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################
        self.down1 = DownConv(kernel =kernel, in_channels = in_channels, out_channels = num_filters , padding=1 )
        self.down2 = DownConv(kernel = kernel, in_channels = num_filters, out_channels = num_filters *2  , padding=1)
        self.bottleneck = Bottleneck(kernel = kernel, in_channels = num_filters*2, out_channels = num_filters *2 , padding = 1)
        self.up1 = UpConv(kernel = kernel, in_channels = num_filters*2, out_channels = num_filters , padding=1)
        self.up2 = UpConv(kernel = kernel, in_channels = num_filters, out_channels = num_colors, padding=1)

        
        
    def forward(self, x):
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################
        #print("in" , x.shape)
        down1_out = self.down1(x)
        #print("down1:",down1_out.shape)
       
        down2_out = self.down2(down1_out)
        #print("down2:",down2_out.shape)

        bottleneck_out = self.bottleneck(down2_out)
        #print("bottel:", bottleneck_out.shape)
        
        skip1 = torch.cat([down2_out , bottleneck_out] , dim = 1)
        #print("skip1:", skip1.shape)
        
        skip_out1 = self.skip_conv1(skip1)
        #print("skip_out1:", skip_out1.shape)
        
        up1_out = self.up1(skip_out1)
        #print("up1:" , up1_out.shape)
        
        skip2 = torch.cat([down1_out , up1_out] , dim = 1)
        skip_out2 = self.skip_conv2(skip2)

        up2_out = self.up2(skip_out2)
        #print("up2:" ,up2_out.shape)
        
        skip3 = torch.cat([x , up2_out] , dim = 1)
        skip_out3 = self.skip_conv3(skip3)
        
        final_output = self.final_conv(skip_out3)
        #print("final:" , final_output.shape)

        return final_output
        
        
class ResidualUNET(nn.Module):
    def __init__(
            self, kernel , num_filters, num_colors, in_channels=1, out_channels=3
    ):
        super(ResidualUNET, self).__init__()


        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        # Other properties if needed
        
        self.final_conv = nn.Conv2d(in_channels = num_colors , out_channels = num_colors , kernel_size = kernel , padding = 1)
        self.skip_conv1 = nn.Conv2d(256 , 128 , 1)
        self.skip_conv2 = nn.Conv2d(128 , 64 , 1)
        self.skip_conv3 = nn.Conv2d(25 , 24 , 1)

        # Down part of the model, bottleneck, Up part of the model, final conv
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################
        self.down1 = DownConv(kernel =kernel, in_channels = in_channels, out_channels = num_filters , padding=1 , residual = True )
        self.down2 = DownConv(kernel = kernel, in_channels = num_filters, out_channels = num_filters *2  , padding=1 , residual = True)
        self.bottleneck = Bottleneck(kernel = kernel, in_channels = num_filters*2, out_channels = num_filters *2 , padding = 1 , residual = True)
        self.up1 = UpConv(kernel = kernel, in_channels = num_filters*2, out_channels = num_filters , padding=1 , residual = True)
        self.up2 = UpConv(kernel = kernel, in_channels = num_filters, out_channels = num_colors, padding=1 , residual = True)

        
        
    def forward(self, x):
        ##############################################################################################
        #                                       Your Code                                            #
        ##############################################################################################
        #print("in" , x.shape)
        down1_out = self.down1(x)
        #print("down1:",down1_out.shape)
        #print("------")
        down2_out = self.down2(down1_out)
        #print("down2:",down2_out.shape)
        #print("------")

        bottleneck_out = self.bottleneck(down2_out)
        #print("bottel:", bottleneck_out.shape)
        #print("------")

        skip1 = torch.cat([down2_out , bottleneck_out] , dim = 1)
        #print("skip1:", skip1.shape)
        #print("------")

        skip_out1 = self.skip_conv1(skip1)
        #print("skip_out1:", skip_out1.shape)
        #print("------")

        up1_out = self.up1(skip_out1)
        #print("up1:" , up1_out.shape)
        #print("------")

        skip2 = torch.cat([down1_out , up1_out] , dim = 1)
        skip_out2 = self.skip_conv2(skip2)

        up2_out = self.up2(skip_out2)
        #print("up2:" ,up2_out.shape)
        #print("------")

        skip3 = torch.cat([x , up2_out] , dim = 1)
        skip_out3 = self.skip_conv3(skip3)
        
        final_output = self.final_conv(skip_out3)
        #print("final:" , final_output.shape)
        #print("------")

        return final_output