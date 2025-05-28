# === Plot Solution in the Plane z=0 
def plot_surface(model, grid_size=100,title='Viscosity Solutions',color='inferno'):
    x = torch.linspace(-1, 1, grid_size)
    y = torch.linspace(-1, 1, grid_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    input_grid = torch.stack([X_flat, Y_flat, torch.zeros_like(X_flat)], dim=1)

    with torch.no_grad():
        U = model(input_grid).cpu().numpy().reshape(grid_size, grid_size)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X.numpy(), Y.numpy(), U, cmap=color)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x, y, 0)')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# === Plot u(x) vs ||x|| ===
def plot_norm_vs_solution(model, d=3, n_samples=1000, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        x_test = sample_unit_ball(n_samples, d, device)
        u_pred = model(x_test)
        u_true = torch.norm(x_test, dim=1, keepdim=True)
        plt.scatter(u_true.cpu(), u_pred.cpu(), alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('True norm ||x||')
        plt.ylabel('Predicted u(x)')
        plt.title('u(x) vs ||x||')
        plt.grid(True)
        plt.show()
        
