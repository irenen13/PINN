
def train_model(model, d=3, h=0.1, lam=100, T=10000, batch_size=1000, eps=1e-4, lr=0.001, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Using device:", device)
    print("Iteration, Loss")

    for i in range(T + 1):
        x = sample_unit_ball(batch_size, d, device)
        x.requires_grad = True

        u = model(x)
        grad_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        grad_norm_sq = torch.sum(grad_u**2, dim=1, keepdim=True)

        laplacian_u = compute_laplacian(u, x)
        # viscosity solutions
        loss_residual = (-eps * laplacian_u + grad_norm_sq - 1)**2
        loss = lam * torch.mean(loss_residual)

        # Boundary loss
        boundary_x = torch.randn_like(x)
        boundary_x = boundary_x / torch.norm(boundary_x, dim=1, keepdim=True)
        u_b = model(boundary_x)
        loss += torch.mean(u_b**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f"{i},{loss.item():.6f}")

    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_network()
model = train_model(model, device=device)
plot_surface(model, grid_size=100,title='Viscosity Solutions',color='inferno')