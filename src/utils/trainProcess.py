import torch
import torch.nn as nn
import torch.optim as optim    

def net_validation(net, valloader, device, epoch, num_epochs, process_name="validation"):
    """
    Função de validação do modelo.
    Args:
        net (nn.Module): O modelo de rede neural.
        valloader (DataLoader): DataLoader com os dados de validação.
        device (torch.device): O dispositivo (CPU ou GPU).
        epoch (int): Número da época atual.
        num_epochs (int): Número total de épocas.
        process_name (str): Nome do processo (validação ou treino).
    """
    net.eval()  # Coloca o modelo em modo de avaliação
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():  # Desativa o cálculo do gradiente
        for data in valloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)  # Passa as entradas pela rede
            loss = criterion(outputs, labels)  # Calcula a perda

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # Obtém as previsões
            total += labels.size(0)  # Acumula o total de exemplos
            correct += (predicted == labels).sum().item()  # Acumula acertos

    # Cálculo da precisão e perda de validação
    accuracy = 100 * (correct / total)
    avg_loss = running_loss / len(valloader)
    
    print(f"Época {epoch+1}/{num_epochs}, Perda de {process_name}: {avg_loss:.4f}, Precisão de {process_name}: {accuracy:.2f}%\n")
    
def trainning_and_val_process(trainloader, valloader, net, device, lr=0.01, momentum=0.9, num_epochs=10):
    """
    Função de treinamento e validação do modelo.
    Args:
        trainloader (DataLoader): DataLoader com os dados de treinamento.
        valloader (DataLoader): DataLoader com os dados de validação.
        net (nn.Module): O modelo de rede neural.
        device (torch.device): O dispositivo (CPU ou GPU).
        lr (float): Taxa de aprendizado.
        momentum (float): Momento para o otimizador.
        num_epochs (int): Número de épocas para o treinamento.
    """
    criterion = nn.CrossEntropyLoss()  # Função de perda
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)  # Otimizador

    for epoch in range(num_epochs):
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
        avg_train_loss = running_loss / len(trainloader)
        print(f"Época {epoch+1}/{num_epochs}, Perda de Treinamento: {avg_train_loss:.4f}")
        
        # Verificar precisão do treino
        net_validation(net, trainloader, device, epoch, num_epochs, process_name="treino")
        # Verificar precisão da validação
        net_validation(net, valloader, device, epoch, num_epochs, process_name="validação")

    print('Treinamento Concluído')
