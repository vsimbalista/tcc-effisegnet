# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:41:31 2024
@author: vitor
"""

#%% Load for Inference
# If you only need the model for testing or inference:
    
model = Net(model=YourModel, criterion=YourCriterion, ...)  # Replace with your Net initialization
checkpoint = torch.load("checkpoints/epoch=360-step=15162.ckpt")
model.load_state_dict(checkpoint["state_dict"])
model.eval()  # Set the model to evaluation mode

#%% Load Entire Trainer
# If you want to resume training from this checkpoint:
    
trainer = L.Trainer(resume_from_checkpoint="checkpoints/epoch=360-step=15162.ckpt")

#%% Testing the Loaded Model
# Once loaded, you can pass test images to the model:
    
test_image = torch.tensor(your_image_data)  # Replace with your test image tensor
test_image = test_image.unsqueeze(0)  # Add batch dimension if needed
output = model(test_image)

#%% Verify Checkpoint Contents
# To inspect the contents of the checkpoint file:
    
checkpoint = torch.load("checkpoints/epoch=360-step=15162.ckpt")
print(checkpoint.keys())  # View all keys in the checkpoint

