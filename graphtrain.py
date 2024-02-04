import matplotlib.pyplot as plt


# Function to parse a line and extract losses
def parse_line(line):
    parts = line.split()
    epoch = int(parts[1])
    sens_loss = float(parts[4])
    contrastive_loss = float(parts[7])
    edge_recon_loss = float(parts[11])
    feature_recon_loss = float(parts[15])
    return epoch, sens_loss, contrastive_loss, edge_recon_loss, feature_recon_loss


# Function to read the file and parse losses
def read_losses(file_path):
    epochs = []
    sens_losses = []
    contrastive_losses = []
    edge_recon_losses = []
    feature_recon_losses = []

    with open(file_path, 'r') as file:
        for line in file:
            epoch, sens_loss, contrastive_loss, edge_recon_loss, feature_recon_loss = parse_line(line)
            epochs.append(epoch)
            sens_losses.append(sens_loss)
            contrastive_losses.append(contrastive_loss)
            edge_recon_losses.append(edge_recon_loss)
            feature_recon_losses.append(feature_recon_loss)

    return epochs, sens_losses, contrastive_losses, edge_recon_losses, feature_recon_losses


# Function to plot the losses
def plot_losses(epochs, sens_losses, contrastive_losses, edge_recon_losses, feature_recon_losses):
    plt.figure(figsize=(10, 8))
    plt.plot(epochs, sens_losses, label='Sensitivity Loss')
    plt.plot(epochs, contrastive_losses, label='Contrastive Loss')
    plt.plot(epochs, edge_recon_losses, label='Edge Reconstruction Loss')
    plt.plot(epochs, feature_recon_losses, label='Feature Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('NBA Training Losses over Epochs with Graphair')
    plt.legend()
    plt.show()


# Main function to run the script

epochs, sens_losses, contrastive_losses, edge_recon_losses, feature_recon_losses = read_losses("training.txt")
plot_losses(epochs, sens_losses, contrastive_losses, edge_recon_losses, feature_recon_losses)
