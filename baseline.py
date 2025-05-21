import torch, time
from torchvision import transforms, datasets
import os

def main():
    # disable OpenMP threads to prevent interference with pytorch multiprocessing
    os.environ['OMP_NUM_THREADS'] = '1'

    # Define transformations
    transform = transforms.Compose([ #defines pipeline
        transforms.RandomRotation(30),  # data augmentation
        transforms.ToTensor(),          # convert PIL images to PyTorch tensors, normalize [0,1]
    ])
    
    # load CIFAR-10 dataset(training set 50,000 images 32x32rgb) (will download automatically if not present)
    #saves dataset to ./data directory, uses training split, download data set if missing, applies defined transformation
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    print(f"Dataset size: {len(dataset)} samples") 
    
    # Windows-specific multiprocessing settings
    #prevent memory issues whrn sharing data processes
    torch.multiprocessing.set_sharing_strategy('file_system')
    #ensures stable mutiprocessing on windows
    torch.multiprocessing.set_start_method('spawn', force=True)


    for workers in [0, 2, 4, 8]:
        # config data loader
        loader = torch.utils.data.DataLoader(
            dataset,                   # dataset not subdata set
            batch_size=64,  #processes 64 images at a time
            num_workers=workers,  
            shuffle=True,  #randomizes data order, making it realistic as possible
            persistent_workers=workers>0  # reduce process creation overhead
        )
        
        # Warm-up phase (critical for accurate timing)
        if workers > 0: #warm up for multiprocessing 
            next(iter(loader))  # init workers since first batch will be slower
        
        start = time.perf_counter()
        
    
        for epoch in range(3):
            epoch_start = time.perf_counter() #start timer
            for images, _ in loader: #iterate through batches, ignored batches will be labeled _
                images.mean().item()    # force computation, preventing optimiation
            epoch_time = time.perf_counter() - epoch_start #epoch duration
            print(f"workers={workers}, Epoch={epoch+1}, Time={epoch_time:.2f}s")
        
        # Calculate total time
        total_time = time.perf_counter() - start
        print(f"workers={workers}, Total Time={total_time:.2f}s (Avg {total_time/3:.2f}s/epoch)\n")

if __name__ == "__main__":
    main()  
#expectation: 0 to be slower bc it's 1 batch at a time+ cpu argumentation. gpu will wait for data in cases
#              8(parallel) to be faster bc mutiple CPU workers prefeetch batches, reeduces GPU idle time because there's mutiple batches where 1 can load faster than another.
    
#the goal of baseline benchmark is to compare the performance of Pytorch's DataLoaer with differen num_workers settings from 0 vs 8(Pytorch default speed b4 optimizations)
#measure how much parallel data loading(>0) would speed up the data preprocessing compared to sequential loading(=0) when training a deep learning model

#a. find the most optimal setting for given system. b. find impact of data loaing on training speed.(see if the time difference is significantly diff.)
# NOTE: if there isn't a sig. diff in speed then it can mean CPU bottlenect(too few cores) and/or ineffienct transforms(randomrotation can be optimized)