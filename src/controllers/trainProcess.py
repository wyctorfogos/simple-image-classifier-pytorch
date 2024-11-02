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
            if i % 100 == 99:  # Print every 100 batches
                print(f"[Época {epoch + 1}, Lote {i + 1}] Perda média do lote: {running_loss / 100}")
                running_loss = 0.0  # Reset running_loss for the next set of batches

    print('Finished Training')