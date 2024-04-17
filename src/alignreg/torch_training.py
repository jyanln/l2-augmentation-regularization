import numpy as np
import torch
from tqdm import tqdm

from .regularization import torch_maxsql2
from .augment import torch_default_augmentations

loss_history = []
accuracy = []

def measure_acc(model, testloader, num_batch=-1):
    '''
    The function provides the accuracy of the model given the parameters.
    
    Attrs:
    -----
    model: the model passed defined by the user for training
    testloader: Dataloader for testing data
    num_batch: number of batches to be used
    '''

    correct = 0
    total = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):
            if num_batch > 0 and i >= num_batch:
                break

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

def torch_train(train_loader,
                test_loader,
                model, 
                optimizer, 
                epochs, 
                loss_fn,
                l2_lambda,
                record_batch_size=1,
                acc_batch=-1,
                lazy_augmentations=[]):
    '''
    The function train the model if written by user in Tensorflow
    and return the loss and the accuracy of the model given the 
    parameters
    
    Attrs:
    -----
    aug_loader: pytorch dataloader for AugmentedDataset (see augment.py)
    model: the model passed defined by the user for training
    optimizer: optimizer used by the model while training 
    loss_fn: loss function used by the model while training
    l2_lambda: The lambda parameter for regularizing on sql2 difference
    record_batch_size: Batch size for recording data
    lazy_augmentations: if specified, uses the given augmentation functions
                        instead of preprocessed augmented images
    '''
    accuracy.clear()
    loss_history.clear()

    running_loss = 0.0

    # training
    for epoch in tqdm(range(epochs)):

        running_loss = 0.0
        for i, (images, aug_images, label) in enumerate(train_loader):
            optimizer.zero_grad()


            logits = model(images)
            loss_val = loss_fn(logits, label)

            # process augmentations if needed
            if lazy_augmentations:
                aug_images = list(map(lambda transform: transform(images), lazy_augmentations))


            reg = torch_maxsql2(model, images, aug_images, loss_fn, label, l2_lambda)
            loss_val = torch.add(loss_val, reg)
            loss_val.backward()
            optimizer.step()

            running_loss += loss_val.item() / record_batch_size
            if i % record_batch_size == record_batch_size - 1:
                loss_history.append(running_loss)
                running_loss = 0.0

        accuracy.append(measure_acc(model, test_loader, acc_batch))

    return accuracy[:], loss_history[:]
