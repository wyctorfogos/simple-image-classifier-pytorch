import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim    

def trainning_and_val_process(trainloader, valloader, net, device, lr=0.01, momentum=0.9, num_epochs=10):

    # Escolha da função Loss e do otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    
    for epoch in range(num_epochs):
        # Modo de treinamento
        net.train()  # Coloca o modelo em modo de treinamento
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Zera os gradientes

            outputs = net(inputs)  # Passa as entradas pela rede
            loss = criterion(outputs, labels)  # Calcula a perda
            loss.backward()  # Calcula os gradientes
            optimizer.step()  # Atualiza os pesos

            running_loss += loss.item()
        
        # Média da perda durante a época
        print(f"Época {epoch+1}/{num_epochs}, Perda de Treinamento: {running_loss / len(trainloader):.4f}")

        # Validação
        net.eval()  # Coloca o modelo em modo de avaliação
        correct = 0
        total = 0
        
        with torch.no_grad():  # Desativa o cálculo do gradiente
            for data in valloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)  # Passa as entradas pela rede
                _, predicted = torch.max(outputs.data, 1)  # Obtém as previsões
                total += labels.size(0)  # Acumula o total de exemplos
                correct += (predicted == labels).sum().item()  # Acumula acertos

        # Cálculo da precisão de validação
        accuracy = 100 * correct / total
        print(f"Época {epoch+1}/{num_epochs}, Precisão de Validação: {accuracy:.2f}%\n")

    print('Finished Training')
