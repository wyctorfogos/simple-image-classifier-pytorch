import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim    

def trainning_process(trainloader, net, device ):

    ## Escolha da função Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    
    running_loss = 0.0
    for epoch in range(10):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data # Separar os dados em entrada e saída
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Época {epoch}: {running_loss}\n")
    print('Finished Training')