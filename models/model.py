import torch
import torch.nn as nn
import torch.nn.functional as F


class WideBranchNet(nn.Module):

    def __init__(self, time_length=7, num_classes=[127, 8]):
        super(WideBranchNet, self).__init__()
        
        self.time_length = time_length
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),

            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(self.time_length, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.3)
            )
        self.max2d = nn.MaxPool2d(2, 2)
        self.classifier_1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes[0])
        )
        self.classifier_2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes[1])
        )
        
    
    def forward(self, x):
        out = self.model(x)
        out = out.squeeze(2)
        out = self.max2d(self.conv2d(out))
        out = out.view((out.size(0), -1))
        out1 = self.classifier_1(out)
        out2 = self.classifier_2(out)
        return out1, out2



class WideBranchNetClips(nn.Module):
    def __init__(self, num_clips=5, time_length=7, num_classes=[127, 81]):
        super(WideBranchNetClips, self).__init__()
        
        self.num_clips = num_clips
        self.time_length = time_length
        self.num_classes = num_classes

        # Conv3D layers to process the stacked clips
        self.model = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(self.num_clips, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.3)
        )
        self.max2d = nn.MaxPool2d(2, 2)

        # Predictors for temporal and spatial tasks
        self.classifier_temporal = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes[0])  # Temporal task output
        )
        self.classifier_spatial = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes[1])  # Spatial task output
        )

    def forward(self, x):
        """
        Forward pass for Clips Net.
        Input shape: (batch_size, 1, num_clips * time_length, H, W)
        """
        out = self.model(x)  # (batch_size, 64, 1, H', W')
        out = out.squeeze(2)  # Remove temporal dimension reduced to 1
        out = self.max2d(self.conv2d(out))  # (batch_size, 64, H'', W'')
        out = out.view((out.size(0), -1))  # Flatten for fully connected layers

        # Temporal and Spatial predictions
        out_temporal = self.classifier_temporal(out)
        out_spatial = self.classifier_spatial(out)
        return out_temporal, out_spatial


if __name__ == '__main__':
    net_clips = WideBranchNetClips(num_clips=5, time_length=7, num_classes=[127, 81])
    test_input_clips = torch.rand(2, 1, 5 * 7, 64, 64)  # Batch size 2, Clips size
    temporal_out, spatial_out = net_clips(test_input_clips)
    print("Temporal Output Shape:", temporal_out.shape)
    print("Spatial Output Shape:", spatial_out.shape)

    # WideBranchNet Test
    net = WideBranchNet(time_length=7, num_classes=[49, 81])
    test_input = torch.rand(2, 1, 7, 64, 64)
    temporal_out, spatial_out = net(test_input)
    print("Temporal Output Shape (Single Clip):", temporal_out.shape)
    print("Spatial Output Shape (Single Clip):", spatial_out.shape)