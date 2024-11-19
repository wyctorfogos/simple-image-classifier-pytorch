import torch
import torch.nn as nn
import torch.optim as optim    

def net_validation(net, valloader, device, epoch, num_epochs, process_name="validation"):
    net.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in valloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * (correct / total)
    avg_loss = running_loss / len(valloader)

    print(f"Época {epoch+1}/{num_epochs}, Perda de {process_name}: {avg_loss:.4f}, Precisão de {process_name}: {accuracy:.2f}%\n")

    return avg_loss, accuracy  # Retorna para possível registro ou gráficos


def trainning_and_val_process(trainloader, valloader, net, device, lr=0.01, momentum=0.9, num_epochs=10, save_path="./src/results/model.pth"):
    net.to(device)  # Move o modelo para o dispositivo
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)  # Reduz LR a cada 5 épocas

    best_val_acc = 0.0  # Para rastrear a melhor precisão de validação

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(trainloader)
        print(f"Época {epoch+1}/{num_epochs}, Perda de Treinamento: {avg_train_loss:.4f}")

        # Validação
        train_loss, train_acc = net_validation(net, trainloader, device, epoch, num_epochs, process_name="treino")
        val_loss, val_acc = net_validation(net, valloader, device, epoch, num_epochs, process_name="validação")

        # Checkpoint para o melhor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc}, save_path)
            print(f"Modelo salvo com precisão de validação: {val_acc:.2f}%")

        scheduler.step()  # Atualiza a taxa de aprendizado

    print('Treinamento Concluído')
